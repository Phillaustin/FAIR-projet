import torch
import torchvision
import pandas
from sklearn.preprocessing import LabelBinarizer

"""
Télécharger le dataset csv "Adult income dataset" sur kaggle 
https://www.kaggle.com/wenruliu/adult-income-dataset
"""

data = pandas.read_csv("adult.csv")
"""
Les paramètres :
'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'

On prédit le genre

"""
y = torch.Tensor(LabelBinarizer().fit_transform(data.gender)).squeeze()





print()