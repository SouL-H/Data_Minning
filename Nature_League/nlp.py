# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk
import re
yorumlar = pd.read_csv('Restaurant_Reviews.csv')


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

cikti = []  
for i in range(1000):
    yorum= re.sub('[^a-zA-Z]',' ' ,yorumlar['Review'][i]) #Review kolonunun i.elemanını alıyoruz
    #a-z A-Z yazmamızdaki sebep bu harflerin haricindekileri almayacağız.
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum =[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    #corpustan stopwordsları atıp takılıları köklerine çevirip listeye atıyor
    yorum = ' '.join(yorum) # join birleştirme işlemidir.Yorumu alıp boşlukla birleştirdik.Ve Str çevirdik.
    # Şimdi işlenebilir.
    cikti.append(yorum)

# Doğal Dil işleme
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000) # en çok kullanılan kelimeyi 1000 tane ile sınırlandır.

X=cv.fit_transform(cikti).toarray()
Y=yorumlar.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size =0.20,random_state=0) # 1000 verinin 800 işle 200 tanesini tahmin et

from sklearn.naive_bayes import GaussianNB # Bu kütüphaneyi kullanıyoruz.
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
Y_pred =gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,Y_pred)
print(cm) # %72.5 civarında başarı gösterdi. %27.5 civarında hata payımız var.
