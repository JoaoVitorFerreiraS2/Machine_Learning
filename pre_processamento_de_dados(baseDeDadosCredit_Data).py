import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Vamos trabalhar com duas base dados


base_credit = pd.read_csv('credit_data.csv')
#=======================================================================#
#ClientID = ID do usuário
#Income = Valor que o usuário recebe por ano
#Age = Idade do usuário
#Loan = Divida que o usuário possui
#Default = Se está devendo [1]: não pagou o empréstimo | [0]: pagou o empréstimo
#=======================================================================#

# Exploração de Dados
#=======================================================================#
#.tail = Ele conta do último até o primeiro (<Valor das linhas que você quer>)
#.head = Ele conta do primeiro até o último (<Valor das linhas que você quer>)
#.describe = Statisticas
#=======================================================================#

#=======================================================================#
#Exemplos do tail e head:
#print(base_credit.tail())
print(base_credit.describe())
print('-=-'*20)
#Exemplo de filtro
#print(base_credit[base_credit['income'] >= 20014.489470])
#=======================================================================#

#=======================================================================#
#Usando a biblioteca numpy
#.unique é referente a contagem única
print(np.unique(base_credit['default'], return_counts=True))
#=======================================================================#

# Visualização dos dados
#=======================================================================#
#Usando a bilioteca seaborn para geração de graficos com a biblioteca matplotlib
#seaborn é usado para a criação do gráfico 
#matplotlib é para mostrar o gráfico
#palette para colocar cores diferentes | Exemplo: palette=('#1f77b4', '#ff7f0e'

#sns.countplot(x = base_credit['default'])
#plt.show()

#.hist separa por números
#plt.hist(x= base_credit['age'])
#plt.hist(x= base_credit['income'])
#plt.hist(x= base_credit['loan'])
#plt.show()
#=======================================================================#

#=======================================================================#
#Usando a biblioteca Plotly
#grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
#grafico.show()
#=======================================================================#

# Tratamento de valores Inconsistentes
#=======================================================================#
#.loc para localizar dados especificos
#O que ocorre é que a base de dados pode ter erros, por exemplo, uma idade negativa, então devemos corrigir isso
#Primeiro localizamos os dados
print(base_credit.loc[base_credit['age'] < 0])
print(base_credit[base_credit['age'] < 0])

#.drop para retirar
#base_credit2 = base_credit.drop('age', axis = 1)

#index = Quantidade de linhas
#base_credit2 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
#print(base_credit2[base_credit2['age'] < 0])

#Mas a forma correta é fazer manualmente para entender o motivo do erro

#.mean pegar a média
print(f'A média das idades é: {base_credit['age'].mean()}')
#Alterando o valor de index especificos
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
print(f'A média das idades atualizadas é: {base_credit['age'].mean()}')

print(base_credit.head(26))
#=======================================================================#

#Tratamento de valores faltantes
#=======================================================================#
#.isnull para verificar se tem valores
#.sum para somar os que são verdadeiros e falsos
print(base_credit.isnull().sum())
print(base_credit.loc[pd.isnull(base_credit['age'])])

#.fillna para preencher
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])
#=======================================================================#

# Divisão entre previsores e classe
#=======================================================================#
#.iloc = serve para selecionar linhas e colunas
#.values = para converter para numpy
x_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

#=======================================================================#

# Escalonamento dos valores
#=======================================================================#
#.min = serve para pegar o menor
#.max = serve para pegar o maior
print(x_credit[:,0].min())
print(x_credit[:,1].max())
scaler_credit = StandardScaler()

#.fit_transform = Transforma os valores para a mesma escala
x_credit = scaler_credit.fit_transform(x_credit)
print(x_credit[:,0].min())
print(x_credit[:,1].max())
#=======================================================================#

# Base de dados do censo
#=======================================================================#

#=======================================================================#