# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:02:19 2020

@author: 25335
"""

# This program outputs BERT HuggingFace as a txt embedding for torchtext processing

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
from torchtext import data
from torchtext.data import Field, BucketIterator
import spacy
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from rouge_score import rouge_scorer
import time
from pytorch_transformers import *
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils
import gensim
from smart_open import open

if torch.cuda.is_available():
    device = torch.device('cuda')
else: device = torch.device('cpu')

#free your gpu memory
torch.cuda.empty_cache()

#spacy_en = spacy.load('en')
# WIN: change
spacy_en = spacy.load(r"C:\Users\25335\anaconda3\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.3.0")
spacy_en = spacy.load('en_core_web_sm')
def tokenize(text):
    return [example.text for example in spacy_en.tokenizer(text)]


#build fields and add sos and eos
SOURCE = data.Field(tokenize = None, #tokenize for exp3
                    pad_token = '[PAD]',
                    init_token = '[CLS]',
                    eos_token = '[SEP]',
                    #unk_token = '[UNK]',#a key error bug in torchtext on adding this 
                    lower = True,
                    fix_length = 2000)
TARGET = data.Field(tokenize = None,#None for BERT 
                    pad_token = '[PAD]',
                    init_token = '[CLS]',
                    eos_token = '[SEP]',
                    #unk_token = '[UNK]',
                    lower = True,
                    fix_length = 64)#should be more

fields = {'source': ('source', SOURCE), 'target': ('target', TARGET)}

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        #path = r'C:\Users\25335\Downloads\summarization_new', #4000
                                        #path = r'C:\Users\25335\Downloads\summarization_new\data\abstractive', # 10k
                                        path = r'C:\Users\25335\Downloads\summarization_new\data', # 10k
                                        train = 'train_WP.json',
                                        validation = 'valid_WP.json',
                                        test = 'test_WP.json',
                                        #train = 'train.json',
                                        #validation = 'valid.json',
                                        #test = 'test.json',
                                        format = 'json',
                                        fields = fields)


#building dictionaries
SOURCE.build_vocab(train_data)
TARGET.build_vocab(train_data)
print('\n')


print(f"{len(train_data.examples)} training examples")
print(f"{len(valid_data.examples)} validation examples")
print(f"{len(test_data.examples)} testing examples")
print('\n')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = 20,
    sort=False,
    device = device)

print(vars(train_data.examples[111])['target'])
print('\n')
#get a BERT dictionary first

# building embeddings for both intput and output
#check tensor from BERT
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# input embedding

'''
d = {} #get a normal word-index dictionary
for i in range(0,len(SOURCE.vocab)):
    #dict_vocab[SOURCE.vocab.itos[i]] = i
    subword = SOURCE.vocab.itos[i]
    inputs = tokenizer(subword, return_tensors="pt")
    outputs = model(**inputs)
    BERT_vec = outputs[1][0][0:128].data.numpy()#use only 3
    d[subword] = BERT_vec
    print(subword)
'''    
    
    

'''
#convert BERT tensors into a normal dict
d = {}
for ind,token in enumerate(tokenizer.vocab.keys()):
    if ind >1995:
        inputs = tokenizer(token, return_tensors="pt")
        outputs = model(**inputs)
        BERT_vec = outputs[1][0][0:128].data.numpy()#use only 128
        d[token] = BERT_vec
'''

'''
# using gensim on BERT caused decoding errors in torch.vocab; might be useful in other projects
m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=3)
m.vocab = d
m.vectors = np.array(list(d.values()))

def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary and row.shape[0] == 128:#128 dim
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

#save the emb
my_save_word2vec_format(binary=True, fname='BERT_128D_S.vec', total_vec=len(d), vocab=m.vocab, vectors=m.vectors)#or save as .bin

#m2 = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('train3d.vec', binary=True)
# convert to txt for testing
#from gensim.models.keyedvectors import KeyedVectors
#model = KeyedVectors.load_word2vec_format('train3d.vec', binary=True)
#model.save_word2vec_format('train3d.txt', binary=False)

'''



'''
# SOURCE txt vector
f=open('128.txt','w',encoding='utf-8')
for i in range(0,len(SOURCE.vocab)):
    subword = SOURCE.vocab.itos[i]
    inputs = tokenizer(subword, return_tensors="pt")
    outputs = model(**inputs)
    BERT_vec = outputs[1][0][0:128].tolist()
    BERT_string = ' '.join([str(elem) for elem in BERT_vec])
    string = subword + " " + BERT_string + '\n'
    f.write(string)
f.close()
'''

# FULL BERT txt vector
f=open('BERT_128D.txt','w',encoding='utf-8')
for ind,subword in enumerate(tokenizer.vocab.keys()):
    inputs = tokenizer(subword, return_tensors="pt")
    outputs = model(**inputs)
    BERT_vec = outputs[1][0][0:128].tolist()
    BERT_string = ' '.join([str(elem) for elem in BERT_vec])
    string = subword + " " + BERT_string + '\n'
    f.write(string)
f.close()

