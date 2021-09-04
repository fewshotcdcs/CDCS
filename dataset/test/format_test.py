import os
import json
import numpy as np
import random
import re

examples = []
langs = ['SQL', 'Solidity']

def format_test():
    
    for lang in langs:
        FILE_DATA_DIR='./{}.txt'.format(lang)
        OUTPUT_DIR = './{}'.format(lang)

        with open(FILE_DATA_DIR, "r", encoding="utf-8") as f:
                datas = f.readlines()
        
        l = len(datas)
        for idx in range(0, l):
            doc_token = datas[idx].split("<CODESPLIT>")[-2]
            #tmp = 0
            for idx_ in range(0, l):
                dd = datas[idx_]
                #tmp = tmp + 1
                code_token = dd.split("<CODESPLIT>")[-1][:-1]
                m = re.compile(r'/\*.*?\*/', re.S)
                result = re.sub(m, ' ', code_token)
                result = ' '.join(result.split())
                example = (str(1), "URLA", "URLB", doc_token, result)
                example = '<CODESPLIT>'.join(example)
                examples.append(example)
            #print("tmp: {}".format(tmp))
        data_path = OUTPUT_DIR
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_0.txt')
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))
        f.close()

if __name__ == '__main__':
    format_test()
