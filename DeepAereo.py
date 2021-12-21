import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, f1_score
import functions as f

#Carregando dados
ocorr_otnave = f.limpenza_dados()

#Análise exploratória dos dados
analise_dados = f.eda(ocorr_otnave)

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
plt.savefig('OcorrClass.png')
plt.show()

#Encoding das variáveis categóricas para inserção nos modelos preditivos
le = preprocessing.LabelEncoder()
for coluna in ocorr_otnave.colunas:
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



#Usando técnicas de manipulação de dados para otimizar o 'rendimento' do dataset,
# uma vez que o mesmo tem poucos dados armazenados para treino e teste dos modelo de classificação

#Base de dados original
# X_train, X_test, Y_train, Y_test, caso_test  = f.tratamento_dados(X,Y)

#Base de dados otimizada/balanceada
oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, Y_over = oversample.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test, caso_test = f.tratamento_dados(X_over, Y_over)

#Aplicação dos classificadores
modelAereo , history = f.autoencoder(X_train, X_test, Y_train, Y_test)
#modelAereo ,history = f.LSTM_GRU(X_train, X_test, Y_train, Y_test)

#Usando o modelo salvo para gerar matriz confução e outras métricas de avaliação
modelAereo = load_model('model_seqs2.h5')
print(f'Min Loss:{np.max(history["accuracy"])}')

y_pred = modelAereo .predict(X_test)
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
caso_test = modelAereo .predict(caso_test)
caso_test = tf.argmax(caso_test, axis=1)
print("\n"+"__________________________________________________")
if caso_test == 0:
    print ('A Classificação da ocorrencia para o caso teste é:  Acidente')
elif caso_test == 1:
    print ('A Classificação da ocorrencia para o caso teste é:  Incidente')
else:
    print('A Classificação da ocorrencia para o caso teste é:  Incidente Grave')



