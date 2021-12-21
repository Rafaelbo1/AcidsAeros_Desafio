import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, f1_score


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
#print("*** values: ", ocorr_otnave.isin(['***']).sum())
print("Missing values: ", ocorr_otnave.isnull().sum())


#Contagem de cada tipo/label de dado em cada coluna do dataset
for column in ocorr_otnave.columns:
    print("\n" + column)
    print(ocorr_otnave[column].value_counts())

#Plot da contagem e plot das variáveis a serem preditas no modelo de classificação proposto
incid = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'INCIDENTE']
incid_graves = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'INCIDENTE GRAVE']
acids = ocorr_otnave.loc[ocorr_otnave['ocorrencia_classificacao'] == 'ACIDENTE']

fig = plt.figure()
sns.countplot(y=ocorr_otnave.ocorrencia_classificacao, data=ocorr_otnave)
plt.xlabel('Número de ocorrências')
plt.ylabel('Tipo de ocorrência')
plt.title('QUANTIDADE DE OCORRÊNCIAS POR CLASSIFICAÇÃO')
plt.show()

#Encoding das variáveis categóricas para inserção nos modelos preditivos
le = preprocessing.LabelEncoder()
for coluna in ocorr_otnave.columns:
    if (ocorr_otnave[coluna].dtypes == 'object'):
        ocorr_otnave[coluna] = le.fit_transform(ocorr_otnave[coluna])
#Aplicação da Correlação de Pearson para verificar a existência de correlação entre as variáveis
corr = ocorr_otnave.corr()
a=ocorr_otnave.iloc[1,:]
print(ocorr_otnave.iloc[1,:])
print(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
plt.show()

#Separação das variáveis de entrada e da variável de saída para serem inseridas nos modelos
X, Y = ocorr_otnave.drop(['ocorrencia_classificacao'], axis=1), ocorr_otnave['ocorrencia_classificacao']

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

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

#Usando técnicas de manipulação de dados para otimizar o 'rendimento' do dataset,
# uma vez que o mesmo tem poucos dados armazenados para treino e teste dos modelo de classificação

#Base de dados original
# X_train, X_test, Y_train, Y_test, caso_test  = tratamento_dados(X,Y)

#Base de dados otimizada/balanceada
oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, Y_over = oversample.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test, caso_test = tratamento_dados(X_over, Y_over)

#Aplicação dos classificadores
autoencoder, history = autoencoder(X_train, X_test, Y_train, Y_test)
#LSTM_GRU(X_train, X_test, Y_train, Y_test)

#Usando o modelo salvo para gerar matriz confução e outras métricas de avaliação
autoencoderModel = load_model('model_seqs2.h5')
print(f'Min Loss:{np.max(history["accuracy"])}')

y_pred = autoencoderModel.predict(X_test)
pred_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.argmax(Y_test, axis=1)

print(f'F1_Score:{f1_score(pred_categories , true_categories, average="weighted")}')
cf_matrix = confusion_matrix(pred_categories, true_categories)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')


## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Acidente','Incidente', 'Incidente Grave'])
ax.yaxis.set_ticklabels(['Acidente','Incidente', 'Incidente Grave'])
plt.show()


#Testando o modelo com uma ocorrência do banco de dados para validadar
# a classificação/identificação do tipo de acidente que poderá ocorrer
caso_test = autoencoderModel.predict(caso_test)
caso_test = tf.argmax(caso_test, axis=1)
print("\n"+"__________________________________________________")
if caso_test == 0:
    print ('A Classificação da ocorrencia para o caso teste é:  Acidente')
elif caso_test == 1:
    print ('A Classificação da ocorrencia para o caso teste é:  Incidente')
else:
    print('A Classificação da ocorrencia para o caso teste é:  Incidente Grave')



