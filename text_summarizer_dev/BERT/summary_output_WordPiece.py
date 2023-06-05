#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:52:09 2020

@author: taolin
"""

# open the file and read the data
source_lines = open(r"C:\Users\25335\Downloads\summarization_new\data\test.txt.src", encoding='utf-8').\
read().strip().split('\n')
source_lines = source_lines[0:15000]
target_lines = open(r"C:\Users\25335\Downloads\summarization_new\data\test.txt.tgt.tagged" , encoding='utf-8').\
read().replace("</t>",'').replace("<t>",'').strip('').split('\n')
target_lines = target_lines[0:15000]
'''
#txt output
f=open('summary_mini.txt','w')
for i in range(0,len(target_lines)):
    f.write(source_lines[i]+ '\t'+target_lines[i]+'\n')
f.close()
'''

train_src = source_lines[0:8000]
train_trg = target_lines[0:8000]
val_src = source_lines[8000:9000]
val_trg = target_lines[8000:9000]
test_src = source_lines[9000:10000]
test_trg = target_lines[9000:10000]
#jsonline output for torchtext

# use WordPiece tokenization in BERT to get subwords
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def WP_tokenizer(file):
    new = []
    for item in file:
        item = " ".join(tokenizer.tokenize(item))
        new.append(item)
    return new

train_src = WP_tokenizer(train_src)
train_trg = WP_tokenizer(train_trg)
val_src = WP_tokenizer(val_src)
val_trg = WP_tokenizer(val_trg)
test_src = WP_tokenizer(test_src)
test_trg = WP_tokenizer(test_trg)


import jsonlines
#make a list of dictionaries
def jsonwriter(name,src,trg):
    fp = jsonlines.open(name, 'w')
    for i in range(len(src)):
        dictt={'source':src[i],'target':trg[i]}
        fp.write(dictt)
    fp.close()
jsonwriter('train_WP.json',train_src,train_trg)
jsonwriter('valid_WP.json',val_src,val_trg)
jsonwriter('test_WP.json',test_src,test_trg)