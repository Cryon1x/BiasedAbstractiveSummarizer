import os
from os import path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print(':: Importing modules...')

import re
import math
import matplotlib.pyplot as plt
import string
import gensim
import nltk
import spacy
import pickle
import warnings
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from p_tqdm import p_map
import numpy as np
import pandas as pd
import seaborn as sns
import contractions
print('resolving dependencies...')


from tensorflow.keras.layers import Input, LSTM, Embedding, Attention, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
import warnings
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as B
from tensorflow.keras.callbacks import ModelCheckpoint
from lib.Det_Sent import sentiment 
from lib.attentionlayer_borrowed import AttentionLayer
from tensorflow.python.client import device_lib





train_test_ratio = 0.9

global length
length = 200
max_words = 5000
max_len_summary=20

train = pd.read_csv('data/filttest.csv',nrows=100000)


test=train[-math.floor((1 - train_test_ratio * len(train))):]
train=train[:math.floor(train_test_ratio * len(train))]

str_train = train['Strings']
str_train=str_train.astype(str)

sum_train = train['Summary']
sum_train=sum_train.astype(str)

str_test = test['Strings']
str_test=str_test.astype(str)

sum_test = test['Summary']
sum_test=sum_test.astype(str)

print('initializing tokens...')

# tokenize

str_token = Tokenizer()
sum_token = Tokenizer()
str_token.fit_on_texts(list(str_train))
sum_token.fit_on_texts(list(sum_train))

str_train = str_token.texts_to_sequences(str_train) 
str_test = str_token.texts_to_sequences(str_test) 

sum_train = sum_token.texts_to_sequences(sum_train)
sum_test = sum_token.texts_to_sequences(sum_test)

str_train = pad_sequences(str_train,  maxlen=length, padding='post') 
str_test = pad_sequences(str_test, maxlen=length, padding='post')

sum_train = pad_sequences(sum_train,  maxlen=length, padding='post') 
sum_test = pad_sequences(sum_test, maxlen=length, padding='post')

str_voc = len(str_token.word_index) + 1
sum_voc = len(sum_token.word_index) + 1

print('initializing LSTM encoder...')
# LSTM encoder 
dim = 200
enc_inputs = Input(shape=(length)) 
enc_embed = Embedding(str_voc, dim,trainable=True)(enc_inputs)


lstm1 = LSTM(dim, return_sequences=True, return_state=True) 
enc1, h1, c1 = lstm1(enc_embed) 

lstm2 = LSTM(dim, return_sequences=True, return_state=True) 
enc2, h2, c2 = lstm2(enc1)

lstm3 = LSTM(dim, return_sequences=True, return_state=True) 
enc3, h3, c3 = lstm3(enc2) 

lstm4 = LSTM(dim, return_sequences=True, return_state=True) 
enc4, h4, c4 = lstm4(enc3) 

lstm5 = LSTM(dim, return_sequences=True, return_state=True) 
enc0, h0, c0 = lstm5(enc4)


print('initializing LSTM decoder...')
# decoder 
dec_in = Input(shape=(None,)) 
dec_embed_layer = Embedding(sum_voc, dim, trainable=True) 
dec_embed = dec_embed_layer(dec_in) 
dec_lstm = LSTM(dim, return_sequences=True, return_state=True) 
dec_out,dec_fwd, dec_back = dec_lstm(dec_embed,initial_state=[tf.convert_to_tensor(h0), tf.convert_to_tensor(c0)])

# this isn't sequential, so we have to manually concatenate layers. Using an attention layer found on github.
attention_layer = AttentionLayer(name='scanner') 
attention_output, attention_state = attention_layer([enc0, dec_out]) 

dec_in_conc = Concatenate(axis=-1, name='conc_layer')([dec_out, attention_output])


dense = TimeDistributed(Dense(sum_voc, activation='relu')) 
dec_out = dense(dec_in_conc) 


model = keras.models.load_model("SUMM.hdf5")
