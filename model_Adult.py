import torch
import torch.nn as nn
import torchvision
import pandas
from sklearn.preprocessing import LabelBinarizer
import numpy as np

"""
    Télécharger le dataset csv "Adult income dataset" sur kaggle 
    https://www.kaggle.com/wenruliu/adult-income-dataset

    Description de la base : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
"""

data = pandas.read_csv("adult.csv")
"""
Les paramètres :
'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'

On prédit le genre

Sont continues:
    'age', 'fnlwgt', 'educational-num',
    'capital-gain', 'capital-loss', 'hours-per-week'

On encode one-hot : 
    workclass, relationship, race, native-country, occupation, marital-status

On binarise :
    income et gender

On n'utilise pas "education" mais sa version en continue avec educational-num
"""
y = torch.Tensor(LabelBinarizer().fit_transform(data.gender)).squeeze()

data_continues = data[['age', 'fnlwgt', 'educational-num',
                       'capital-gain', 'capital-loss', 'hours-per-week']].to_numpy()

data_one_hot = pandas.get_dummies(data[['workclass', 'relationship', 'race', 'native-country', 'occupation', 'marital-status']]).to_numpy()
data_binary  = LabelBinarizer().fit_transform(data.income)
dataset      = np.concatenate((data_continues, data_binary, data_one_hot), axis = 1)
dataloader   = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)


INPUT_SIZE  = dataset.shape[1]
OUTPUT_SIZE = y.shape[1]
"""
    Création du Séléctionneur, 
    on met qu'une seule couche pour l'instant et on sort en sigmoid pour avoir des proba
"""
H1_SIZE = 100

selector = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE)   , nn.ReLU(),
    nn.Linear(H1_SIZE   , INPUT_SIZE), nn.Sigmoid()
)

"""
    Création du Prédicteur
"""
H1_SIZE = 50
H2_SIZE = 30
predicteur = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1_SIZE)    , nn.ReLU(),
    nn.Linear(H1_SIZE   , H2_SIZE)    , nn.ReLU(),
    nn.Linear(H2_SIZE   , OUTPUT_SIZE), nn.Sigmoid()
)

"""
Optimisation
"""
NB_ITERATION = 100
for i in range(NB_ITERATION):
    for b in 