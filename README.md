<a href='https://github.com/paulozip/aeronautic-occurrences'><b>If you are English spoken, check out my English version of this repo.</b></a>

# Acidentes aéreos no Brasil
Nesta apresentação, eu estarei utilizando dados abertos disponibilizados pela CENIPA - Centro de Investigação e Prevenção de Acidentes aeronáuticos. Tais arquivos conterão informações sobre ocorrências envolvendo aeronaves nos últimos 10 anos. Você pode acessar dados mais atualizados visitando <a href='http://dados.gov.br/dataset/ocorrencias-aeronauticas-da-aviacao-civil-brasileira'>a página oficial de Dados Abertos Brasileiros</a>, mas, caso deseje, poderá estar realizando o download dos datasets utilizados aqui através dos links de download abaixo.
<br>
<br>
Para este estudo, utilizarei de dois datasets da CENIPA:
<br>
<br>
<a href='./dataset/ocorrencia.csv'>Ocorrencia.csv</a>: possui os dados sobre cada ocorrência registrada nos últimos 10 anos. Código da ocorrência, Data, Motivo da Ocorrência e Localização serão encontrados nesse conjunto de dados.
<br>
<br>
<a href='./dataset/aeronave.csv'>Aeronave.csv</a>: informações agrupadas sobre as aeronaves envolvidas nas ocorrências registradas no arquivo ocorrencia.csv. Aqui serão encontrados dados como: Modelo da Aeronave, Tipo de Aeronave, Fabricante, Quantidade de Fatalidades, dentre outras.


# Requisitos

* Python 3.6

As demais ferramentas e libs utilizadas estão no "requirements.txt"
