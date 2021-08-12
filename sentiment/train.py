

def bilstm():

lay = int(input(':: How many layers for the LSTM? (20 recommended) '))
        
# LSTM IMPLEMENTATION
op = 'l'

LSTM = Sequential()
LSTM.add(layers.Embedding(max_words, 40, input_length=max_len))
LSTM.add(layers.Bidirectional(layers.LSTM(lay,dropout=0.65)))
LSTM.add(layers.Dense(3,activation='softmax'))
LSTM.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
his_LSTM = LSTM.fit(x_train, y_train, epochs=35,validation_data=(x_test, y_test),callbacks=[ModelCheckpoint("LSTM_PARAMETERS.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto',save_weights_only=False)])

