# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:40:41 2017

@author: lming
"""

import os
import sys
import numpy as np
from queryStrategy import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dense, Input, Flatten,Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding,GRU, SimpleRNN,GlobalAveragePooling1D, LSTM
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.models import load_model

from keras.utils import plot_model
from keras.layers.core import Layer  
from keras import regularizers, constraints  
from keras import backend as K
from Attention import AttentionWithContext
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FOLDER='reviews_1'
TEXT_DATA_DIR = rootdir+'/datadir/'+DATA_FOLDER+'/target/'
W2V_FILE='reviewsw2v.txt'
GLOVE_DIR = rootdir+'/datadir/word_vec/'+W2V_FILE

QUERY='Random'
EPISODES=10
timesteps=20
numofsamples=5
#BUDGET=timesteps*numofsamples
resultname=DATA_FOLDER+'_'+QUERY+'_result.txt'
'''
rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FOLDER=sys.argv[1]
TEXT_DATA_DIR = rootdir+'/datadir/'+DATA_FOLDER+'/target/'
W2V_FILE=sys.argv[2]
GLOVE_DIR = rootdir+'/datadir/word_vec/'+W2V_FILE


EPISODES=int(sys.argv[3])
timesteps=int(sys.argv[4])
numofsamples=int(sys.argv[5])
QUERY=sys.argv[6]
#BUDGET=timesteps*numofsamples
resultname=DATA_FOLDER+'_'+QUERY+'_result.txt'
'''
# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')

embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# second, prepare text samples and their labels
print('Processing text dataset')
texts=[] #list of text samples
labels_index={} # # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path=os.path.join(TEXT_DATA_DIR,name)
    if os.path.isdir(path):
        label_id=len(labels_index)
        labels_index[name]=label_id
        for fname in sorted(os.listdir(path)):
            if len(fname)>0:
                fpath=os.path.join(path,fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t=f.read().rstrip().strip().replace('\n', ' ').replace('\t', '')
                t = t.replace('1s', '').replace('1n', '')
                t = t.lower()
                texts.append(t)
                f.close()
                labels.append(label_id)
print('Found %s texts' % len(texts))
#print(texts)

#finally, vectorize the text samples into a 2D integer tensor
tokenizer=Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#print(sequences)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_la = np.concatenate((data[:100], data[-100:]), axis=0)
y_la = np.concatenate((labels[:100], labels[-100:]), axis=0)

x_un=data[100:-100]
y_un=labels[100:-100]


print('Preparing embedding matrix.')
# prepare embedding matrix
num_words=max(MAX_NB_WORDS,len(word_index))
embedding_matrix=np.zeros((num_words+1,EMBEDDING_DIM))
for word, i in word_index.items():
    if i>MAX_NB_WORDS:
        continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


allaccuracylist=[] 
for r in range(0, EPISODES):
    accuracylist=[]
    x_trn = x_la[:10]
    y_trn = y_la[:10]

    x_val = x_la[10:]
    y_val = y_la[10:]

    x_pool = x_un
    y_pool = y_un
    print('Repetition:'+str(r+1))
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions    
    embedding_layer=Embedding((num_words+1), EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
    sequence_input=Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences=embedding_layer(sequence_input)
    embedded_sequences=Dropout(0.3)(embedded_sequences)
    #embedded_sequences=SimpleRNN(50, return_sequences=True)(embedded_sequences)
    embedded_sequences=Conv1D(50, 3, activation='relu')(embedded_sequences)
    
    #doc=AttentionWithContext()(embedded_sequences)
    doc=GlobalAveragePooling1D()(embedded_sequences)
    doc=Dense(2)(doc)
    preds=Activation('softmax')(doc)
    #preds=Dense(len(labels_index),activation='softmax', activity_regularizer=regularizers.l2(0.01))(doc)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['acc'])

    querydata=[]
    querylabels=[]
    model.fit(x_trn, y_trn, verbose=0)
    
    querydata = querydata + list(x_trn)
    querylabels = querylabels + list(y_trn)
    print('Model initialized...')

    for t in range(0,timesteps):
        print('Repetition:'+str(r+1)+' Iteration '+str(t+1))
        print('Number of current samples:'+str((t+1)*numofsamples))
        sampledata=[]
        samplelabels=[]
        if(QUERY is 'Random'):
            sampledata, samplelabels, x_pool, y_pool= randomSample(x_pool, y_pool, numofsamples)
        if(QUERY is 'Uncertainty'):
            sampledata, samplelabels, x_pool, y_pool= uncertaintySample(x_pool, y_pool, numofsamples, model)
        if(QUERY is 'Diversity'):
            sampledata, samplelabels, x_pool, y_pool= diversitySample(x_pool, y_pool, numofsamples, querydata)
        querydata=querydata+sampledata
        querylabels=querylabels+samplelabels
        
        x_train = np.array(querydata)
        y_train = np.array(querylabels)
         
        model.fit(x_train, y_train,batch_size=5, epochs=1, verbose=0)
        '''
        if((t+1) % 5 == 0):
            accuracy = model.evaluate(x_val, y_val, verbose=0)[1]
            accuracylist.append(accuracy)
            print(accuracy)
        '''
        accuracy = model.evaluate(x_val, y_val, verbose=0)[1]
        accuracylist.append(accuracy)
        print(accuracy)
    allaccuracylist.append(accuracylist)

accuracyarray=np.array(allaccuracylist)
averageacc=list(np.mean(accuracyarray, axis=0))
print('Accuray list: ')
print(averageacc)
ww=open(resultname,'w')
ww.writelines(str(line)+ "\n" for line in averageacc)
ww.close()