import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def limpenza_dados():
    # Leitura dos CSV's
    ocorrencia = pd.read_csv('ocorrencia.csv', sep=';', low_memory=False)
    ocorrencia_tipo = pd.read_csv('ocorrencia_tipo.csv', sep=';')
    aeronave = pd.read_csv('aeronave.csv', sep=';')

    # Tratamento inicial dos dados
    ocorrencia_tipo = ocorrencia_tipo.iloc[:, :-2]
    aeronave = aeronave.drop(['aeronave_operador_categoria'], axis=1)

    ocorrencia[['Dia', 'Mes', 'Ano']] = ocorrencia.ocorrencia_dia.str.split("/", expand=True, )
    ocorrencia[['hora']] = ocorrencia.ocorrencia_hora.str.split(":", expand=True, )[0]

    ocorr_ot = pd.merge(ocorrencia, ocorrencia_tipo, how='inner', on='codigo_ocorrencia1')
    ocorr_otnave = pd.merge(aeronave, ocorr_ot, how='inner', on='codigo_ocorrencia2')
    ocorr_otnave = ocorr_otnave.drop(['ocorrencia_dia', 'ocorrencia_hora', 'codigo_ocorrencia', 'codigo_ocorrencia1',
                                      'codigo_ocorrencia2', 'codigo_ocorrencia3', 'codigo_ocorrencia4',
                                      'aeronave_pmd_categoria', 'ocorrencia_pais'], axis=1)

    print("Missing values: ", ocorr_otnave.isnull().sum())
    print("__________________________________________________")
    print("*** values: ", ocorr_otnave.isin(['***']).sum())

    # Limpeza de Dados - remover colunas com mais do que 4%(0,4) de dados ausentes
    # Remove as linhas com os dados restantes ausentes
    for coluna in ocorr_otnave:
        print(coluna)
        contarMiss = ocorr_otnave[coluna].isna().sum()  # + ocorr_otnave[coluna].isin(['***']).sum(axis=0)\
        # + ocorr_otnave[coluna].isin(['****']).sum(axis=0)
        perc = (contarMiss / len(ocorr_otnave))
        # coluna com
        if perc > 0.04:
            ocorr_otnave.drop([coluna], axis=1, inplace=True)
        else:
            # ocorr_otnave = ocorr_otnave[ocorr_otnave[coluna] != '***']
            ocorr_otnave = ocorr_otnave[ocorr_otnave[coluna].notna()]

    print("__________________________________________________")
    # print("*** values: ", ocorr_otnave.isin(['***']).sum())
    print("Missing values: ", ocorr_otnave.isnull().sum())
    return ocorr_otnave



def eda (ocorr_otnave):
    #Ocorrências de Incidentes, Incidentes Graves e Acidentes por Ano
    incids = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'INCIDENTE']
    incids_graves = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'INCIDENTE GRAVE']
    acids = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'ACIDENTE']

    fig = plt.figure(figsize=(27,12))
    #fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plot1 = plt.subplot(2, 3, 1)
    ax = sns.countplot(x=acids.Ano, data=acids)
    plt.xlabel("Ano")
    plt.ylabel("Contagem acidentes")
    plt.title("Ocorrências de Acidentes por ano")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 2))

    plot2 = plt.subplot(2, 3, 2)
    ax = sns.countplot(x=incids.Ano, data=incids)
    plt.xlabel("Ano")
    plt.ylabel("Contagem incidentes")
    plt.title("Ocorrências de Incidentes por ano")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 3))

    plot3 = plt.subplot(2, 3, 3)
    ax = sns.countplot(x=incids_graves.Ano, data=incids_graves)
    plt.xlabel("Ano")
    plt.ylabel("Contagem incidentes graves")
    plt.title("Ocorrências de Incidentes Graves por ano")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x() + 0.2, p.get_height() + 1))
    plt.savefig('OcorrAno.png')
    plt.show()

    #Ocorrências por Estado
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(x=ocorr_otnave.ocorrencia_uf, data=ocorr_otnave)
    plt.xlabel("Estado")
    plt.ylabel("Contagem acidentes/incidentes")
    plt.title("Ocorrências por Estado")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x(), p.get_height() + 10))
    plt.savefig('OcorrEstado.png')
    plt.show()

    #Classificação das ocorrências por Tipo de Veículo
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(x=ocorr_otnave.aeronave_tipo_veiculo, data=ocorr_otnave,
                       hue=ocorr_otnave.ocorrencia_classificacao)
    plt.legend(fontsize='x-large')
    plt.xlabel("Tipo de Veículo")
    plt.ylabel("Contagem Classificação")
    plt.title("Classificação das ocorrências por Tipo de Veículo")
    ax.legend(loc='upper right')
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x() + 0.05, p.get_height() + 10))
    plt.savefig('OcorrVeiculo.png')
    plt.show()



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
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(input_dim, activation='sigmoid',
                    activity_regularizer=regularizers.l1(10e-50))(input_layer)
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
                              epochs=400,
                              batch_size=32,
                              shuffle=True,
                              validation_data=(X_test, Y_test),
                              verbose=1,
                              callbacks=[checkpointer]).history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title("AUTOENCODER")
    plt.savefig('AUTOENCODER_Overfitting.png')
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.h5',
                                   verbose=1,
                                   save_best_only=True)
    model.summary()
    history = model.fit(trainX, Y_train,
                        epochs=400,
                        batch_size=32,
                        validation_data=(testX, Y_test),
                        verbose=1,
                        callbacks=[checkpointer]).history

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title("LSTM")
    plt.savefig('LSTM.png')
    plt.show()
    return model,history