
from __future__ import print_function

import os
import sys
import numpy as np
import gensim
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from sklearn.utils import class_weight
import math
from collections import Counter

#BASE_DIR = '/Users/Daywatch'
#GLOVE_DIR = os.path.join(BASE_DIR, 'Downloads/glove.6B')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, 'Downloads/20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

#class_weight: gonna need it in the diss
cw={0:1419,1:14433}
#cw ={0: 2764,1: 252,2: 3890,3:784,4:76,5:1252,6:928,7:659,8:158,9:2403,10:202,11:486,12:307,13:272}
#cw={0: 3016,1: 6930,2:3220,3:688,4:307,5:272}
#cw={0:3328,1:12524}
#cw={0:15659,1:193}

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

#emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/Daywatch/Desktop/summer18/kerasp/eng/GoogleNews-vectors-negative300.bin.gz', binary=True)
#emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/Daywatch/Desktop/summer18/kerasp/sgns.merge.word', binary=False)
emb = gensim.models.KeyedVectors.load_word2vec_format('/projects/tali5971/python_libs/lib/python3.5/site-packages/myresearch/embedding/sgns.merge.word', binary=False)
#emb = gensim.models.KeyedVectors.load_word2vec_format('/projects/tali5971/python_libs/lib/python3.5/site-packages/myresearch/embedding/sgns.merge.char', binary=False)


#print('Found %s word vectors.' % len(embeddings_index))
print('Found %s word vectors.' % len(emb.vocab))

# second, prepare text samples and their labels
print('Processing text dataset')

#add="/Users/Daywatch/Desktop/summer18/kerasp/VDCdiss/exp1_char_l.txt"
add="/projects/tali5971/python_libs/lib/python3.5/site-packages/myresearch/DISS/data2/exp1_seg_l.txt"
f = [line.strip().split("###") for line in codecs.open(add)]
texts=[item[1].split() for item in f]
#exp1
label_index={'NON-VDC':0,'VDC':1}
#exp2:fine
#label_index={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13}
#exp2:coarse
#label_index={'SM':0, 'CM':1, "R":2, 'AS':3, '13':4, '14':5}
#exp3
#label_index={'NON':0,'MET':1}
#exp4
#label_index={'NON':0,'CO':1}
labels=[label_index[item[0]] for item in f]

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

label = to_categorical(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', label.shape)

# split the data into a training set and a validation set

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
num_test_samples=int(TEST_SPLIT * data.shape[0])

x_test=data[:num_test_samples]
y_test=label[:num_test_samples]
y_test=np.asarray([[np.nonzero(item == 1)[0][0]] for item in y_test])

x_val=(data[num_test_samples:])[:num_validation_samples]
y_val=(label[num_test_samples:])[:num_validation_samples]
#y_val=np.asarray([[np.nonzero(item == 1)[0][0]] for item in y_val])

x_train=(data[num_test_samples:])[num_validation_samples:]
y_train=(label[num_test_samples:])[num_validation_samples:]
#y_train=np.asarray([[np.nonzero(item == 1)[0][0]] for item in y_train])

#07/2019try new weights
#A:
'''
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

'''
#C
def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}
cw=get_class_weights(cw, smooth_factor=0.1)


print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
'''
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    #embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
'''

for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
        if word in emb:
            embedding_matrix[i] = emb[word]
        #else:
            #embedding_matrix[i] = np.zeros(EMBEDDING_DIM)
    #embedding_vector = embeddings_index.get(word)
    #if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        #embedding_matrix[i] = embedding_vector

print('Training model.')
model = Sequential()
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
model.add(Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
#new
#model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=128,
          #class_weight=cw,
          class_weight=cw,
          epochs=50,
          validation_data=(x_val, y_val))
#save the model and predict new data!
model.save('lstm_model_exp1.h5')
# make predictions; convert one hot back to list
yhat = model.predict(x_test, verbose=0)
pred = [np.argmax(element) for element in yhat]
y_test = [num[0] for num in y_test]

from sklearn.metrics import recall_score, f1_score, precision_score
#evaluation1: both
p=precision_score(y_test, pred,average=None)
f=f1_score(y_test, pred,average=None)
r=recall_score(y_test, pred,average=None)
print("precision is " +str(p)+"\n") 
print("recall is " +str(r))
print("f1 is " +str(f)+"\n")

#evaluation2: weighted
p=precision_score(y_test, pred,average='weighted')
f=f1_score(y_test, pred,average='weighted')
r=recall_score(y_test, pred,average='weighted')
print("precision is " +str(p)+"\n") 
print("recall is " +str(r))
print("f1 is " +str(f)+"\n")

#write the files of pred and gold 
file=open("pred.txt",'w')
for e in pred:
    file.write(str(e)+"\n")
file.close()
file=open("gold.txt",'w')
for e in y_test:
    file.write(str(e)+"\n")
file.close()