#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values 
y = dataset.iloc[:, 4].values
#Userid veri setimizde gereksiz olduğu için X ile 2 ve 3 sütunu
#Y ile 4.sütunu aldık 


# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Verimizi %75- %25  boyutunda ayırdık.

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
''' 
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı 
Elimizdeki 400 veriyi 100 tanesini tahmine 300 tanesini eğitime ayırmıştık.
CV 4 dememizin sebebi eğitime ayırdığımız verileride 100 şekilde tahmin için işleyecek.
k-ford_cross_validation algoritmasınnı mantığı bu şekilde çalışıyor.

'''
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print(basari.mean())#Basarimizin ortalamasini gösterir 1 ne kadar yakında başarımız o kadar iyidir.
print(basari.std())#Basarimizin standart sapmasini gösterir ne kadar düşükse o kadar başarımız iyidir.









