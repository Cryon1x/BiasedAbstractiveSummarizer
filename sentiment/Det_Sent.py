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
from keras import backend as K
from keras.callbacks import ModelCheckpoint


def sentiment(string,token,max_len,model):
	
    print('Working...')
    


    sentiment = ['Neutral','Negative','Positive']

    sequence = token.texts_to_sequences([string])
    test = pad_sequences(sequence, maxlen=max_len)
    print(sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]])
    print(np.around(model.predict(test), decimals=0).argmax(axis=1)[0])


    #     loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    # print('Accuracy: ',accuracy)
    # predictions = model.predict(x_test)

