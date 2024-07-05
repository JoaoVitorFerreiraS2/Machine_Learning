import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

base_census = pd.read_csv('census.csv')
print(base_census)
print(base_census.isnull().sum())
print(base_census.describe())

#grafico = px.treemap(base_census, path=['workclass', 'age'])
#grafico.show()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

label_encoder_teste = LabelEncoder()

#teste = label_encoder_teste.fit_transform(x_census[:, 1])

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

#print(f'Aqui: {x_census}')

oneHotEncoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = oneHotEncoder_census.fit_transform(x_census).toarray()

#print(f'Aqui: {x_census.shape}')
#print(x_census.shape)


scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)
print(f'Aqui: {x_census}')
#print(x_census.shape)

# Base de Treinamento de dados
# importamos o from sklearn.model_selection import train_test_split
#Crição de 4 váriasveis
#Sem o random_state sempre vai mudar os resultados
x_census_treinamneto, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size = 0.25, random_state=0)

print(x_census_treinamneto.shape)
print(y_census_treinamento.shape)

print(x_census_teste.shape)
print(y_census_teste.shape)

with open ('census.pkl', mode = 'wb') as f:
    pickle.dump([x_census_treinamneto, y_census_treinamento, x_census_teste, y_census_teste])
    