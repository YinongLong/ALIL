# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 15:05:23 2017

@author: lming
"""
import os
import sys
import copy
import keras
import numpy as np
from someutils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Flatten,Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding,GRU, SimpleRNN,GlobalAveragePooling1D, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model
from keras.layers.core import Layer  
from keras import regularizers, constraints  
from keras import backend as K
import time
start_time = time.time()

from Attention import AttentionWithContext
from queryStrategy import *
EMBEDDING_DIM = 40
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FOLDER='AP17'
TEXT_DATA_DIR = rootdir+'/datadir/'+DATA_FOLDER+'/en/'
W2V_FILE='ap17w2v.txt'
GLOVE_DIR = rootdir+'/datadir/word_vec/'+W2V_FILE

EPISODES=2
BUDGET=10
k_num=5

policyname=DATA_FOLDER+'en_policy.h5'
classifiername=DATA_FOLDER+'en_classifier.h5'
'''
rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FOLDER=sys.argv[1]
TEXT_DATA_DIR = rootdir+'/datadir/'+DATA_FOLDER+'/source/'
W2V_FILE=sys.argv[2]
GLOVE_DIR = rootdir+'/datadir/word_vec/'+W2V_FILE

EPISODES=int(sys.argv[3])
BUDGET=int(sys.argv[4])
k_num=int(sys.argv[5])
'''
# first, build index mapping words in the embeddings set
# to their embedding vector
#print('Indexing word vectors.')
embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
#print('Processing text dataset')
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
#print('Found %s texts' % len(texts))

#finally, vectorize the text samples into a 2D integer tensor
tokenizer=Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
del texts
#data set for inisialize the model
x_la = np.concatenate((data[:100], data[-100:]), axis=0)
y_la = np.concatenate((labels[:100], labels[-100:]), axis=0)

x_un=data[100:-100]
y_un=labels[100:-100]

#print('Preparing embedding matrix.')
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

print('Begin training active learning policy..')
#load random initialised policy
policy=getPolicy(k_num,104)     
for tau in range(0,EPISODES):
    #Shuffle D_L
    indices = np.arange(len(x_la))
    np.random.shuffle(indices)
    x_la=x_la[indices]
    y_la=y_la[indices]
    #Split initial train,  validation set
    x_trn=x_la[:10]
    y_trn=y_la[:10]
    
    x_val=x_la[10:]
    y_val=y_la[10:]
    
    x_pool=list(x_un)
    y_pool=list(y_un)
    #Initilize classifier
    model= getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
    model.fit(x_trn, y_trn,batch_size=1,epochs=1, verbose=0)
    model.save(classifiername)
    #Memory (two lists) to store states and actions
    states=[]
    actions=[]
    #toss a coint
    coin=np.random.rand(1)
    #In every episode, run the trajectory
    for t in range(0,BUDGET):
        print('Episode:'+str(tau+1)+' Budget:'+str(t+1))
        x_new=[]
        y_new=[]
        accuracy=0
        row=0
        #save the index of best data point or acturally the index of action
        bestindex=0 
        #Random sample k points from D_pool
        x_rand_unl, y_rand_unl, queryindices = randomKSamples(x_pool, y_pool, k_num)
        for datapoint in zip(x_rand_unl, y_rand_unl):
            #model_temp = load_model(classifiername)
            model_temp= getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)
            model_temp.fit(x_trn, y_trn,batch_size=1,epochs=1, verbose=0)
            x_temp=datapoint[0]
            y_temp=datapoint[1]
            x_temp_trn= np.expand_dims(x_temp, axis=0)
            y_temp_trn=np.expand_dims(y_temp, axis=0)
            
            history= model_temp.fit(x_temp_trn, y_temp_trn, validation_data=(x_val, y_val),batch_size=1, epochs=1, verbose=0)
            val_accuracy = history.history['val_acc'][0]
            if(val_accuracy>accuracy):
                bestindex=row
                accuracy=val_accuracy
                x_new=x_temp
                y_new=y_temp
            row=row+1
            del model_temp
        state=getAState(x_trn, y_trn,x_rand_unl,model)
        # if head(>0.5), use the policy; else tail(<=0.5), use the expert
        if(coin>0.5):
            #tempstates= np.ndarray((1,K,len(state[0])), buffer=np.array(state))
            tempstates= np.expand_dims(state, axis=0)
            action=policy.predict_classes(tempstates)[0]
        else:
            action=bestindex
        states.append(state)
        actions.append(action)
        x_trn=np.vstack([x_trn,x_new])
        y_trn=np.vstack([y_trn,y_new])
        model.fit(x_trn, y_trn, batch_size=1, epochs=1, verbose=0)
        model.save(classifiername)
        
        index_new=queryindices[action]
        del x_pool[index_new]
        del y_pool[index_new]
        
    states=np.array(states)
    actions=to_categorical(np.asarray(actions))
    del model
    del x_trn
    del y_trn
    del x_val
    del y_val
    del x_pool
    del y_pool
    #policy=getPolicy(states.shape[1], states.shape[2])
    policy.fit(states,actions)

policy.save(policyname)
print('Policy saved.')
print("--- %s seconds ---" % (time.time() - start_time))
del policy

