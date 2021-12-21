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

# Requisitos

* Python 3.6

As demais ferramentas e libs utilizadas estão no "requirements.txt"
<br>
<br>

# Análise exploratória dos dados
<br>
<br>
<br>
Foi realizado o tratamento inicial para limpeza e organização dos dados a fim de gerar a visualização de gráficos para melhor compreensão do dataset utilizado.
Optou-se por remover as colunas com mais do que 4% de dados ausentes, tendo em vista o tigo de dado(categórico) e a maior complexidade de realizar técnicas de input para "missing values". Para as colunas que ainda ficaram com 'rows' sem dado, optou-se por excluir toda a linha para manter a integridade da base de dados.
<br>
<br>
Foram realizadas algumas análises na base de dados do desafio para compreender melhor como os dados estão "distribuidos". A partir de então, percebeu-se claramente um desbalanceamento no que foi nossa variável de saída dos modelos preditivos que foram extraidas da coluna "classificacao_ocorrencia": Aciente, Incidente e Incidente Grave. Com tal desbalanceamento pode haver problemas de overfitting, maior aprendizado para a variável de maior quantidade de dados. Tal hipótese se confirmou nos teste realizados.



 
