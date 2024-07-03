import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

base_census = pd.read_csv('census.csv')
print(base_census)
print(base_census.isnull().sum())
print(base_census.describe())