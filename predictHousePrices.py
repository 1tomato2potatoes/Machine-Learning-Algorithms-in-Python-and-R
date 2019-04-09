# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:42:27 2019

@author: anh.vnl
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
# matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = 12,10

# Importing the dataset
dataset = pd.read_csv('22_workshop_data.csv')

# Drop features if percentage of missing values in that feature is greater than 20%
null_dataset = dataset.columns[dataset.isnull().any()]

for col in null_dataset:
    if round((dataset[col].isnull().sum())/1460, 2) > 0.2:
        dataset.drop(col, axis=1, inplace=True)
        
# Handle ordinal data columns
# creating a dict file for lot shape 
lotShape = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0} 
# Mapping lot shape data 
dataset.LotShape = [lotShape[item] for item in dataset.LotShape] 

# creating a dict file for Utilities
utilities = {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}
# Mapping utilities data 
dataset.Utilities = [utilities[item] for item in dataset.Utilities] 

# creating a dict file for Land Slope
landSlope = {'Gtl': 2, 'Mod': 1, 'Sev': 0}
# Mapping Land Slope data 
dataset.LandSlope = [landSlope[item] for item in dataset.LandSlope] 

# creating a dict file for exter quality
quality = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
# Mapping Exter Qual data 
dataset.ExterQual = [quality[item] for item in dataset.ExterQual] 
# Mapping Exter Cond data 
dataset.ExterCond = [quality[item] for item in dataset.ExterCond] 
# Mapping HeatingQC data 
dataset.HeatingQC = [quality[item] for item in dataset.HeatingQC]
# Mapping KitchenQual data 
dataset.KitchenQual = [quality[item] for item in dataset.KitchenQual]

# creating a dict file for basement quality
bmstQuality = {'Ex': 10, 'Gd': 9, 'TA': 8, 'Fa': 7, 'Po': 6, 'Other': 0}
# Mapping Bsmt Qual data 
dataset.BsmtQual.fillna('Other', inplace=True)
dataset.BsmtQual = [bmstQuality[item] for item in dataset.BsmtQual] 
# Mapping Bsmt Cond data 
dataset.BsmtCond.fillna('Other', inplace=True)
dataset.BsmtCond = [bmstQuality[item] for item in dataset.BsmtCond]
# Mapping Garage Qual data 
dataset.GarageQual.fillna('Other', inplace=True)
dataset.GarageQual = [bmstQuality[item] for item in dataset.GarageQual]
# Mapping Garage Cond data 
dataset.GarageCond.fillna('Other', inplace=True)
dataset.GarageCond = [bmstQuality[item] for item in dataset.GarageCond]

# creating a dict file for Bsmt Exposure
bsmtExposure = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'Other': 0}
# Mapping Bsmt Qual data 
dataset.BsmtExposure.fillna('Other', inplace=True)
dataset.BsmtExposure = [bsmtExposure[item] for item in dataset.BsmtExposure]

# creating a dict file for BsmtFin Type 1
bsmtFinType = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'Other': 0}
# Mapping BsmtFin Type 1 data 
dataset.BsmtFinType1.fillna('Other', inplace=True)
dataset.BsmtFinType1 = [bsmtFinType[item] for item in dataset.BsmtFinType1]
# Mapping BsmtFin Type 2 data 
dataset.BsmtFinType2.fillna('Other', inplace=True)
dataset.BsmtFinType2 = [bsmtFinType[item] for item in dataset.BsmtFinType2]

# creating a dict file for Central Air
centralAir  = {'Y': 1, 'N': 0}
# Mapping Central Air data 
dataset.CentralAir = [centralAir[item] for item in dataset.CentralAir]

# creating a dict file for Garage Finish
garageFinish  = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'Other': 0}
# Mapping Garage Finish data 
dataset.GarageFinish.fillna('Other', inplace=True)
dataset.GarageFinish = [garageFinish[item] for item in dataset.GarageFinish]

# creating a dict file for Paved Drive
pavedDrive  = {'Y': 2, 'P': 1, 'N': 0}
# Mapping Paved Drive data 
dataset.PavedDrive = [pavedDrive[item] for item in dataset.PavedDrive]

# creating a dict file for electrical
electrical  = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1, 'Other': 0}
# Mapping Electrical data 
dataset.Electrical.fillna('Other', inplace=True)
dataset.Electrical = [electrical[item] for item in dataset.Electrical]

# creating a dict file for Functional
functional  = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
# Mapping Functional data 
dataset.Functional = [functional[item] for item in dataset.Functional]

# handle mising data of garage year built column
dataset.GarageYrBlt.fillna(0, inplace=True)
# handle mising data of garage type column
dataset.GarageType.fillna('NoGarage', inplace=True)
# handle mising data of Mas Vnr Area
dataset.MasVnrArea.fillna((dataset.MasVnrArea.mean()), inplace=True)
# handle mising data of Mas Vnr Type
dataset.MasVnrType.fillna((dataset.MasVnrType.mode().iloc[0]), inplace=True)
# handle mising data of Lot Frontage
dataset.LotFrontage.fillna((dataset.LotFrontage.mean()), inplace=True)
# end

# Handle categorical data
cha_columns = dataset.select_dtypes(include=["object"])
#onehotencoder = OneHotEncoder()
for col in cha_columns:  
    dataset = pd.get_dummies(dataset, columns=[col], prefix = [col], drop_first = True)

# Split data
newX=dataset.drop('SalePrice',axis=1)
newY=dataset['SalePrice']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=1403, random_state=0)  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test) 

importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Cross validation for test dataset (10-fold validation)
from sklearn.cross_validation import cross_val_score
rfr_test_score = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Test Mean score:", rfr_test_score.mean())
print("Test Standard Deviation:", rfr_test_score.std())

regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test)
