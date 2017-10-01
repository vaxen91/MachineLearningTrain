# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv('Data.csv')

# x[range row, range column] es x[:,1] tutte le righe della col con indice 1
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#gestire i dati mancanti
#con imputer vado a caricare le colonne con valori null
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
#sostituisce i valori null con il mean (media) della column
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:,0])
#dummy encoding
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()

labelEncoder_y = LabelEncoder()
y[:] = labelEncoder_y.fit_transform(y[:])

#splitting dataset in training set and test set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =0 )

#scaling data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_train)

