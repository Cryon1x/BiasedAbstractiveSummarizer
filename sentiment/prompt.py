


# suppress tf jargon
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import re
import matplotlib.pyplot as plt
import string
import gensim
import nltk
import spacy
import pickle
import warnings
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import seaborn as sns

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
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend 
from keras.callbacks import ModelCheckpoint
from Det_Sent import sentiment 


print('Module imports successful!')


def main():

    # Read data, clean 

    train = pd.read_csv('data/train.csv')
    train = train[['sentiment','selected_text']]

    # train = pd.read_csv(filepath_or_buffer='data/bigtrain.csv', header=None, usecols=[5,0], names=['sentiment','selected_text'],encoding="ISO-8859-1")
    # selected_text = pd.read_csv(filepath_or_buffer='data/bigtrain.csv', header=None, usecols=[6], names=['selected_text'],encoding="ISO-8859-1")

    # train = pd.concat([selected_text, sentiment],axis=1)
    train["selected_text"].fillna("No content", inplace = True)

    print('Importing tokens (this might take a while)...')


    # train = clean(train)

    temp = []

    max_len = 200

    data = train['selected_text'].values.tolist()


    for i in range(len(data)):
        temp.append(clean(data[i]))
    #print(list(temp[:12]))

    symb_exp = list(d2words(temp))

    temp = []
    for i in range(len(symb_exp)):
        temp.append(TreebankWordDetokenizer().detokenize(symb_exp[i]))

    temp = np.array(temp)
    label = np.array(train['sentiment'])


    det = []
    for i in range(len(label)):
        lab_num = filt(label[i])
        det.append(lab_num)

    det = np.array(det)
    label = tf.keras.utils.to_categorical(det, 3, dtype="float32")

    #print(len(labels))




    # Sequencer (change values of maxes for different performance/speed tradeoffs)

    max_words = 5000


    token = Tokenizer(num_words=max_words)
    token.fit_on_texts(data)
    sequences = token.texts_to_sequences(temp)
    f = open("demofile2.txt", "a")

    strings = pad_sequences(sequences, maxlen=max_len)

    # print('Sample integer word matrix representation and labels:')
    # print(strings)
    # print(label)

    # Full test and train datasets determined
    x_train, x_test, y_train, y_test = train_test_split(strings,label, random_state=0)

    print('Label Checkpoint Passed!')



    if input(':: Proceed with training? [y/N] ') == "y":
    

        lay = int(input(':: How many layers for the LSTM? (20 recommended) '))
        
        # LSTM IMPLEMENTATION
        op = 'l'

        LSTM = Sequential()
        LSTM.add(layers.Embedding(max_words, 40, input_length=max_len))
        LSTM.add(layers.Bidirectional(layers.LSTM(lay,dropout=0.65)))
        LSTM.add(layers.Dense(3,activation='softmax'))
        LSTM.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        his_LSTM = LSTM.fit(x_train, y_train, epochs=35,validation_data=(x_test, y_test),callbacks=[ModelCheckpoint("LSTM_PARAMETERS.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto',save_weights_only=False)])


            # op = 'R'
            # RNN = Sequential()
            # RNN.add(layers.Embedding(max_words, 50))
            # RNN.add(layers.SimpleRNN(20))
            # RNN.add(layers.Dense(3,activation='relu'))


            # RNN.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
            # his_RNN = RNN.fit(x_train, y_train, epochs=5,validation_data=(x_test, y_test),callbacks=[ModelCheckpoint("RNN_PARAMETERS.hdf5", monitor='val_accuracy', verbose=2,save_best_only=True, mode='max',save_weights_only=False)])

    model = keras.models.load_model("LSTM_PARAMETERS.hdf5")
    while True:
        val = input('==> Determine sentiment of input? \n==> ')
        sentiment(val,token,max_len,model)



# clean data was based on a third party source (generic cleaning procedure it seems)

def clean(data):
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    data = re.sub('\S*@\S*\s?', '', data)
    data = re.sub("\'", "", data)
    data = re.sub('\s+', ' ', data)
    return data

def d2words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def filt(x):
    if x == 'negative':
            return 1
    if x == 0:
            return 1
    if x == 'neutral':
            return 0
    if x == 2:
            return 0
    if x == 'positive':
            return 2
    if x == 4:
            return 2

main()

