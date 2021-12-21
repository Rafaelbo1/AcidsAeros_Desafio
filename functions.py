import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


#Função de preparação dos dados com 'padronização' dos valores entre 0 e 1,
# divisão do dataset em treino e teste
# Cadificação com one-hot-encoding da variável de saida do modelo
def tratamento_dados(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    Y_test = to_categorical(Y_test)
    Y_train = to_categorical(Y_train)

    caso_test = sc.transform(np.array([[
        2953.0, 2.0, 35.0, 345.0, 75.0, 5.0, 1.0, 4377.0, 8.0,
        1979.0, 3.0, 3.0, 2.0, 10.0, 325.0, 325.0, 8.0, 7.0,
        3.0, 0.0, 116.0, 11.0, 2.0, 0.0, 0.0, 1.0,
        0.0, 4.0, 1.0, 0.0, 12.0, 77.0]]))
    return X_train, X_test, Y_train, Y_test, caso_test

#As DeepLearning à seguir foram programadas para serem salvass
# a fim de evitar a reexecução do código em ambiente de teste e produção

# CRIANDO UMA DEEPLEARNING AUTOENCODER
def autoencoder(X_train, X_test, Y_train, Y_test):
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    nb_epoch = 10
    batch_size = 32
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(input_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-50))(input_layer)
    encoder = (Dense(64, activation='sigmoid')(encoder))
    encoder = Dense(64, activation='sigmoid')(encoder)
    # encoder = Dense(86, activation='sigmoid')(encoder)
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(output_dim, activation='softmax')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="model_seqs2.h5",
                                   verbose=0,
                                   save_best_only=True)

    history = autoencoder.fit(X_train, Y_train,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, Y_test),
                              verbose=1,
                              callbacks=[checkpointer]).history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    return autoencoder,history



# DEEP LEARNING LSTM MODULÁVEL PRA GRU - BASTA SUBISTITUI DIRETO NO CÓDIGO UMA PELA OUTRA
def LSTM_GRU(X_train, X_test, Y_train, Y_test):
    # preparando os dados
    input_dim = X_train.shape[1]
    trainX = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Criando uma LSTM DeepLearning
    model = Sequential()
    model.add(LSTM(40, input_shape=(trainX.shape[1], 1)))
    # model.add(LSTM(20))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.h5',
                                   verbose=1,
                                   save_best_only=True)
    model.summary()
    history = model.fit(trainX, Y_train, epochs=300, batch_size=32,
                        validation_data=(testX, Y_test), verbose=1,
                        callbacks=[checkpointer]).history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    return model,history