import argparse
import glob
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)


from transformers import AutoTokenizer

from utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

import higher
import torch.optim as optim


logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.logging_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, checkpoint=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info('loss %s', str(tr_loss - logging_loss))
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + idx))

            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(last_output_dir)
            logger.info("Saving model checkpoint to %s", last_output_dir)
            idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            with open(idx_file, 'w', encoding='utf-8') as idxf:
                idxf.write(str(args.start_epoch + idx) + '\n')

            torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

            step_file = os.path.join(last_output_dir, 'step_file.txt')
            with open(step_file, 'w', encoding='utf-8') as stepf:
                stepf.write(str(global_step) + '\n')

            if (results['acc'] > best_acc):
                best_acc = results['acc']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def maml_train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.logging_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    

    for lang, _ in train_dataset.items():
        logger.info("  Training Language = %s", lang)

    # Train!
    logger.info("***** Running MAML training *****")
    #logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    #logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()

    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        n_inner_iter = 1
        inner_opt = torch.optim.SGD(model.parameters(), lr = 1e-4)

        for lang, dataset in train_dataset.items():
            logger.info("  Training Language = %s", lang)

            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            task_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            task_dataloader = DataLoader(dataset, sampler=task_sampler, batch_size=int(args.train_batch_size))
            t_total = len(task_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
            logger.info("  Num examples = %d", len(dataset))
            logger.info("  Total optimization steps = %d", t_total)

            for step, batch in enumerate(task_dataloader):
                
                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                    fast_model.train()
                    set_seed(args)
                    batch = tuple(t.to(args.device) for t in batch)
                    n_meta_lr = int((args.train_batch_size)/2)

                    input_fast_train = {'input_ids': batch[0][n_meta_lr:],
                            'attention_mask': batch[1][n_meta_lr:],
                            'token_type_ids': batch[2][n_meta_lr:] if args.model_type in ['bert', 'xlnet'] else None,
                            'labels': batch[3][n_meta_lr:]}

                    outputs = fast_model(**input_fast_train)
                    loss = outputs[0]
                    #logger.info("  loss = %s", loss.item())
                    diffopt.step(loss)
                
                
                    input_fast_valid = {'input_ids': batch[0][:n_meta_lr],
                                'attention_mask': batch[1][:n_meta_lr],
                                'token_type_ids': batch[2][:n_meta_lr] if args.model_type in ['bert', 'xlnet'] else None,
                                'labels': batch[3][:n_meta_lr]}
                        
                    outputs = fast_model(**input_fast_valid)
                    qry_loss = outputs[0]
                    #logger.info("  qry_loss = %s", qry_loss.item())
                    try:
                        qry_loss.backward()
                    except:
                        logger.info("Error: Backward()")


                    tr_loss += qry_loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        global_step += 1

                if (step + 1) % args.task_batch_size == 0:
                    optimizer.step()
                    inner_opt.zero_grad()
                    model.zero_grad()
                
                if global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', args.learning_rate, global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    
            logger.info("meta-learning it %s", idx)
            logger.info("  loss = %s", tr_loss / global_step)

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(last_output_dir)
                logger.info("Saving model checkpoint to %s", last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(args.start_epoch + idx) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

    return global_step, tr_loss / global_step
        

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if (mode == 'dev'):
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='dev')
        elif (mode == 'test'):
            eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:

                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds_label, out_label_ids)
        results.update(result)
        if (mode == 'dev'):
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write('evaluate %s\n' % checkpoint)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        elif (mode == 'test'):
            output_test_file = args.test_result_dir
            output_dir = os.path.dirname(output_test_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_test_file, "w") as writer:
                logger.info("***** Output test results *****")
                all_logits = preds.tolist()
                for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                    instance_rep = '<CODESPLIT>'.join(
                        [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])

                    writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))

    return results

def evaluate_MAML(args, model, tokenizer, checkpoint=None, prefix="", mode='dev', data_dir=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if (mode == 'dev'):
            eval_dataset = load_and_cache_MAML_examples(args, eval_task, tokenizer, ttype='dev', data_dir=data_dir)
        elif (mode == 'test'):
            eval_dataset, instances = load_and_cache_MAML_examples(args, eval_task, tokenizer, ttype='test', data_dir=data_dir)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Lang = %s", data_dir)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:

                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds_label, out_label_ids)
        results.update(result)
        if (mode == 'dev'):
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write('evaluate %s\n' % checkpoint)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        elif (mode == 'test'):
            output_test_file = args.test_result_dir
            output_dir = os.path.dirname(output_test_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_test_file, "w") as writer:
                logger.info("***** Output test results *****")
                all_logits = preds.tolist()
                for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                    instance_rep = '<CODESPLIT>'.join(
                        [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])

                    writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))

    return results



def load_and_cache_examples(args, task, tokenizer, ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset

def load_and_cache_MAML_examples(args, task, tokenizer, ttype='train', data_dir=""):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset

def load_and_cache_examples_features(args, task, tokenizer, ttype='train', data_dir=""):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, args.test_file)

        features = convert_pre_train_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=1,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    return features

def get_MAML_train_dataset(args, task, tokenizer, langs_list):

    features = []
    for lang in langs_list:
        logger.info("MAML Training Language: %s", lang)
        datadir = os.path.join(args.data_dir, lang)
        f = load_and_cache_examples_features(args, args.task_name, tokenizer, ttype='train', data_dir = datadir)
        features = features + f
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='codesearch', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_meta_train", action='store_true',
                        help="Whether to run training with meta-learning method.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--task_batch_size", default=100, type=int,
                        help="Task batch size for MAML.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_dir', type=str, default="",
                        help="Log dir.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="test file")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
        
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    
    args.tokenizer_name = "roberta-base"

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_meta_train:
        logger.info("DO META LEARNING")
        lang_list = {"php", "python"}
        langs_dataset = {}
        for lang in lang_list:
            datadir = os.path.join(args.data_dir, lang)
            train_dataset = load_and_cache_MAML_examples(args, args.task_name, tokenizer, ttype='train', data_dir = datadir)
            langs_dataset[lang] = train_dataset
            logger.info("Training/evaluation Language: %s", lang)

        langs_dataset_to_maml = {lang: dataset for lang, dataset in langs_dataset.items()}
        global_step, tr_loss = maml_train(args, langs_dataset_to_maml, model, tokenizer, optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer)
        
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if (args.do_train or args.do_meta_train) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict:
        print('testing')
        model = model_class.from_pretrained(args.pred_model_dir)
        model.to(args.device)
        evaluate(args, model, tokenizer, checkpoint=None, prefix='', mode='test')

    return results


if __name__ == "__main__":
    main()
