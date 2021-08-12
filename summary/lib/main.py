
# suppress tf jargon
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


print(device_lib.list_local_devices())


# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
stop_words = set(stopwords.words('english'))
B.clear_session()


def main():



	train_test_ratio = 0.9
	# import the sentiment analysis 
	global length
	length = 60
	max_words = 5000
	max_len_summary=20

	if path.exists('data/filttest.csv'):
		if input(':: Filtered data detected. Generate new train/test data (not recommended)? [y/N] ') == 'y':
			filtfile = pd.DataFrame()
			print('using train/test ratio of ',train_test_ratio)

			train = pd.read_csv('data/train.csv')
			train = train[['sentiment','selected_text']]
			train["selected_text"].fillna("No content", inplace = True)
			data = train['selected_text'].values.tolist()
			print(':: Starting data import...')


			global token
			token = Tokenizer(num_words=max_words)
			token.fit_on_texts(data)


			train=pd.read_csv("data/reviews.csv")

			print('filtering data...')
			

			train["Text"].fillna("No content", inplace = True)
			train = train[['Summary','Text']]

			# clean this monstrosity 
			temp = []
			data = train['Text'].values.tolist()
			origstrings = data
			
			for i in tqdm(range(len(data))):
				temp.append(clean(data[i]))

			datastr = temp

			train['Strings'] = datastr
			
			temp = []
			train["Summary"].fillna("No content", inplace = True)
			data = train['Summary'].values.tolist()

			global LSTMmodel
			LSTMmodel = keras.models.load_model("lib/LSTM_PARAMETERS.hdf5")
			print('loading sentiment LSTM (this might take a while)...')

			# while True:
			# 	val = input('==> Determine sentiment of input? \n==> ')
			# 	sentiment(val,token,length,model)

			for i in tqdm(range(len(data))):
				summ = clean(data[i])
				sentiment(origstrings[i],token,length,LSTMmodel)
				if sentiment(origstrings[i],token,length,LSTMmodel) == 1:
					summ = "bad " + summ      # integrate sentiment analysis into training data
					# print("Summary:", summ)
					# print("Text:", train['Strings'][i])
					# print("\n")
				if sentiment(origstrings[i],token,length,LSTMmodel) == 2:
					summ = "good " + summ      # integrate sentiment analysis into training data
					# print("Summary:", summ)
					# print("Text:", train['Strings'][i])
					# print("\n")
				temp.append(summ)
				# print(temp)



			orig_data = data 
			data = temp


			train['Summary'] = data
			train['Summary'] = train['Summary'].apply(lambda x: '_START_ '+ x + ' _END_')
			pd.DataFrame(train[['Summary','Strings']]).to_csv("data/filttest.csv")




	# split into test and training data 
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


	model = Model([enc_inputs, dec_in], dec_out) 
	model.summary()

	if input(':: Train model? [y/N] ') == 'y':
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		history=model.fit([str_train,sum_train[:,:-1]], sum_train.reshape(sum_train.shape[0],sum_train.shape[1], 1)[:,1:] ,epochs=10,validation_data=([str_test,sum_test[:,:-1]], sum_test.reshape(sum_test.shape[0],sum_test.shape[1], 1)[:,1:]),callbacks=[ModelCheckpoint("SUMM.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto',save_weights_only=False)],batch_size=32)

	model = keras.models.load_model("SUMM.hdf5",custom_objects={'AttentionLayer': AttentionLayer})


	# build inference model to input text 



	global rev_indexsum 
	global rev_indexstr 
	global index 

	rev_indexsum = sum_token.index_word 
	rev_indexstr = str_token.index_word 
	index = sum_token.word_index

	global enc_model
	enc_model = Model(inputs=enc_inputs,outputs=[enc0, h0, c0])

	dec_in_h = Input(shape=(dim,))
	dec_in_c = Input(shape=(dim,))
	dec_o = Input(shape=(length,dim))

	dec_embed0 = dec_embed_layer(dec_in)
	dec_out0, s_h, s_c = dec_lstm(dec_embed0, initial_state=[dec_in_h, dec_in_c])

	attnout, attnstate = attention_layer([dec_o, dec_out0])
	decoder_cat = Concatenate(axis=-1, name='conc_layer')([dec_out0, attnout])
	dec_out0 = dense(decoder_cat)

	global dec
	dec = Model([dec_in] + [dec_o,dec_in_h, dec_in_c], [dec_out0] + [s_h, s_c])



	# for i in range(len(str_test)):
	# # 	print("Original String:",original_summary(str_test[i]))
	# # 	print("Actual summary:",summary(sum_test[i]))
	# 	print("Predicted summary:",decoder(str_test[i].reshape(1,length)))
	# # 	print("")





def summary(input):
    string=''
    for i in input:
      if((i!=0 and i!=index['start']) and i!=index['end']):
        string=string+rev_indexsum[i]+' '
    return string

def original_summary(input):
    string=''
    for i in input:
      if(i!=0):
        string=string+rev_indexstr[i]+' '
    return string


def clean(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    data = re.sub('\S*@\S*\s?', '', data)
    data = re.sub("\'", "", data)
    data = re.sub('\s+', ' ', data)
    data = re.sub(r"'s\b","",data)
    data = re.sub("[^a-zA-Z]", " ", data) 
    data = data.lower()
    data = contractions.fix(data)
    tokens = [x for x in data.split() if not x in stop_words]
    temp=[]
    
    for i in tokens:
        if len(i)>4:                  
            temp.append(i)   
    return (" ".join(temp)).strip()



def senti(origdata,summ):

	if sentiment(origdata,token,length,LSTMmodel) == 1:
		summ = "bad " + summ      # integrate sentiment analysis into training data
			# print("Summary:", summ)
			# print("Text:", train['Strings'][i])
			# print("\n")
	if sentiment(origdata,token,length,LSTMmodel) == 2:
		summ = "good " + summ      # integrate sentiment analysis into training data
		# print("Summary:", summ)
		# print("Text:", train['Strings'][i])
		# print("\n")

		return summ


def decoder(input):
    e_out, e_h, e_c = enc_model.predict(input)
    sampling = np.zeros((1,1))
    sampling[0, 0] = index['start']

    stop_condition = False
    string = ''
    while not stop_condition:
        output_tokens, h, c = dec.predict([sampling] + [e_out, e_h, e_c])
        tindex = np.argmax(output_tokens[0, -1, :])
        sampled_token = rev_indexsum[tindex]

        if(sampled_token!='end'):
            string += ' '+sampled_token
            if (sampled_token == 'end' or len(string.split()) >= (length-1)):
                stop_condition = True
        sampling = np.zeros((1,1))
        sampling[0, 0] = tindex

        e_h, e_c = h, c


    print(string)
    return string



main()


