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

from Attention import AttentionWithContext
from queryStrategy import *
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

rootdir=os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FOLDER=sys.argv[1]
TEXT_DATA_DIR = rootdir+'/datadir/'+DATA_FOLDER+'/source/'
W2V_FILE=sys.argv[2]
GLOVE_DIR = rootdir+'/datadir/word_vec/'+W2V_FILE

EPISODES=sys.argv[3]
BUDGET=sys.argv[4]
k_num=sys.argv[5]

policyname=DATA_FOLDER+'_policy.h5'
classifiername=DATA_FOLDER+'_classifier.h5'

print(TEXT_DATA_DIR)
print(GLOVE_DIR)
print(EPISODES)
print(BUDGET)
print(k_num)
print(policyname)
print(classifiername)