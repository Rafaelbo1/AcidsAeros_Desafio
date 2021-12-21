<a href='https://github.com/Rafaelbo1/AcidsAeros_Desafio'></a>

# Acidentes aéreos no Brasil
Para este desafio foi utilizado parte do dataset que contém dados abertos disponibilizados pela CENIPA - Centro de Investigação e Prevenção de Acidentes aeronáuticos.
Os arquivos utilizados neste estudo foram: 'ocorrencia.csv', 'aeronave.csv' e 'ocorrencia_tipo.csv' que possuem informações das ocorrências e seus tipos e que envolveram variadas aeronaves aeronaves nos últimos 11 anos. É possível acessar dados mais atualizados visitando <a href='https://dados.gov.br/dataset/ocorrencias-aeronauticas-da-aviacao-civil-brasileira'>a página oficial de Dados Abertos Brasileiros</a>, para melhor utilizar as análises e executar os modelos construidos para esse desafio, é recomendado fazer o download dos arquivos através dos links de download abaixo.
<br>
<br>
As Entidades e seus respctivos atribuitos foram extraídos dos seguintes arquivos da CENIPA:
<br>
<br>
<a href='ocorrencia.csv'>Ocorrencia.csv</a>: Dados sobre cada ocorrência registrada nos últimos 11 anos. Dados: Código da ocorrência, Data, Motivo da Ocorrência e Localização.
<br>
<br>
<a href='aeronave.csv'>Aeronave.csv</a>: informações agrupadas sobre as aeronaves envolvidas nas ocorrências registradas no arquivo ocorrencia.csv. Dados: Modelo da Aeronave, Tipo de Aeronave, Fabricante, Quantidade de Fatalidades, etc.
<br>
<br>
<a href='ocorrencia_tipo.csv'>Aeronave.csv</a>: As informações sobre o tipo de ocorrência estão contidas neste aquivo, do qual foi tirada apenas os dados da coluna onde está descrito o tipo da ocorrência registrada em 'ocorrencia.csv'.
<br>

# Requisitos

* Python 3.6
* As demais ferramentas e libs utilizadas estão no "requirements.txt"
<br>

# Arquivos .py
* DeepAereo.py - é o código 'liso' para melhor vizualização do que está sendo feito em cada etapa.
* functions.py - Onde estão inserídas as funções para execução da EDA, LSTM e GRU e Autoencoder, além do carregamento, limpeza e tratamento dos dados.
<br>

# Análise exploratória e aplicação do desafio.
<br>
<br>
Etapa 1 - Foi realizado o tratamento inicial para limpeza e organização dos dados a fim de gerar a visualização de gráficos para melhor compreensão do dataset utilizado.
Optou-se por remover as colunas com mais do que 4% de dados ausentes, tendo em vista o tigo de dado(categórico) e a maior complexidade de realizar técnicas de input para "missing values". Para as colunas que ainda ficaram com 'rows' sem dado, optou-se por excluir toda a linha para manter a integridade da base de dados.
<br>
<br>
Etapa 2 - Foram realizadas algumas análises na base de dados do desafio para compreender melhor como os dados estão "distribuidos". A partir de então, percebeu-se claramente um desbalanceamento no que foi nossa variável de saída dos modelos preditivos que foram extraidas da coluna 'ocorrencia_classificacao': Aciente, Incidente e Incidente Grave. Com tal desbalanceamento pode haver problemas de overfitting, maior aprendizado para a variável de maior quantidade de dados. Tal hipótese se confirmou nos teste realizados.
<br>
<br>
Etapa 3 - Como a base de dados utilizada possui valores categóricos, houve a necessidade de codificação dos dados categóricos. Realizou-se para esta etapa a codificação por "enumeração" das "labels" ou usando a biblioteca 'scikit-learn - LabelEncoder()'. Optou-se por esse método de codificação por ser o mais simple e por exigir menor processamento na execução.
Para a codificação dos dados categóricos de saída, optou-se pelo one-hot-encoding, uma vez que modelos de aprendizagem de máquina para classificação ajustam-se melhor com esse tippo de saída, usando como função 'loss' a loss='categorical_crossentropy' e a métrica de avaliação o modelo a metrics=['accuracy'].
<br>
<br>
Etapa 4 - Acredita-se que com a base de dados é possível contruir um modelo de classificação para a variável 'ocorrencia_classificacao', a partir dessa premissa, realizou-se nesta etapa a utilização de dois algorítmos de DeepLearning (o objetivo era executar um algorítmo mais trabalhoso para maior aplicação de conhecimentos de modelos de IA/machinelearning - posteriormente, pode-se aplica outros como árvore de decisão, regressão logística, etc.).
<br>
<br>
O código do modelo está pronto para usar os algorítmos de LSTM e GRU e um Autoencoder. Ao final temos as métricas de avaliação e um teste com a inserção de uma ocorrência na IA salva, simulando, de forma beeemmm simples, uma aplicação prática, tal simulação é a última execução do código.
<br>
A realização da técnica de balancemanto também foi aplicada, sendo possível a utilização do código com e sem a técnica.
<br>

# Resultados.

* Overfitting - Para os dados desbalanceados houve claramente overfitting. o modelo de treino chegou a uma métrica de acurácia de até 99% enquanto o maxímo do teste foi de 86%. É possível ver o impacto do desbalancemanto principalmente na matriz confusão, onde o modelo tem melhor desempenho/acertividade na classificação daquela variável com maior qunatidade de dados no dataset
* Aplicação de métodos - foram aplicados variadas técnicas para tentar reverter o overfittng como o drop, bacthnormalizatin, inicialização uniforme dos pesos, mudança de função de ativação, entre outras. O método que reduziu drasticamente essa diferença de resultados entre treino e teste foi o de balancear a base de dados usando o método de "replicação" de novos dados com base nas características do dataset para as variáveis de menor quantidade no dataset.
* Resultado Final - o modelo continuou um uma relativa diferença nos resultado. 99% no treino e 94% no teste. É possível que esse resultado seja melhorado com outras configurações dos algorítmos usados e aplicações de novos métodos.
* Simulação - Para simulação utilizou-se uma linha/amostra do dataset em que o resultado da classificação é  Incidente. Percebe-se que a classificação é realizada corretamente.
* Recomendações - Orienta-se utilizar uma base de dade maior, ou usar novas técnicas para aumentar a quantidade de dados tanto para as variáveis com menor quantidade como de um modo geral, pois, tem-se por nova hipótese, que uma maior quantidade de dados irá trazer maior aprendizado da IA.

 
