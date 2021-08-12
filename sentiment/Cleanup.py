

import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
print('Imports Successful')


def main():
    train = pd.read_csv('~/Documents/Git/tweet-sentiment-extraction/train.csv')
    train = train[['selected_text','sentiment']]
    train["selected_text"].fillna("No content", inplace = True)
    temp = []
    #Splitting pd.Series to list
    data_to_list = train['selected_text'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(depure_data(data_to_list[i]))
    #print(list(temp[:12]))

    data_words = list(sent_to_words(temp))

    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))


    data = np.array(data)

    


    labels = np.array(train['sentiment'])
    y = []
    for i in range(len(labels)):
        if labels[i] == 'neutral':
            y.append(0)
        if labels[i] == 'negative':
            y.append(1)
        if labels[i] == 'positive':
            y.append(2)
    y = np.array(y)
    labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
    del y

    #print(len(labels))

    print('Label Checkpoint Passed')


    # Sequencer (change values of maxes for different performance/speed tradeoffs)

    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweets = pad_sequences(sequences, maxlen=max_len)

    print('Sample integer word matrix representation and labels:')
    print(tweets)
    print(labels)

    # Full test and train datasets determined
    X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)
    print (len(X_train),len(X_test),len(y_train),len(y_test))






    # ACTUAL LSTM IMPLEMENTATION

    model1 = Sequential()
    model1.add(layers.Embedding(max_words, 20))
    model1.add(layers.LSTM(15,dropout=0.5))
    model1.add(layers.Dense(3,activation='softmax'))


    model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    #Implementing model checkpoins to save the best metric and do not lose it on training.
    checkpoint1 = ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
    history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint1])



    best_model = keras.models.load_model("best_model.hdf5")
    print('Model accuracy: ',test_acc)
    predictions = best_model.predict(X_test)




def depure_data(data):
    
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
        
    return data


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),     
deacc=True))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        



def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


main()