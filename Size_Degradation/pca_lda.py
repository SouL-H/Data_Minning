#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
veriler = pd.read_csv('Wine_veriseti.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#verimizi beşe bir oranda train ve test olarak ayırdık.

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #verimizi 2 boyuta indirgemek için 2 yazdık

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)#Random_state kullanmamızın sebebi 2 kere LR uygulayacağız aynı sonuçları elde etmemek istedik.
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)#Random state sayesinde üsteki ile aralarında doğacak benzerliği engellemiş olduk.
classifier2.fit(X_train2,y_train)

#tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LDA donusumunden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train) #PCA bir farkı bunda ikinci boyutuda verdik veriyi etiketlemek için ikinci boyut gereklidir.

#LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

from sklearn.metrics import confusion_matrix #Matrix üzerinden sonuçlarımızı görmemizi sağlayan kütüphane
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

#LDA sonrası / orijinal 
print('lda ve orijinal')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)#Çıkan sonuçlar sonucu en verimli LDA olduğu görünmektedir aralarında.Verileri etiketlemek için 2.girdi vermemizin bu başarıda payı büyük.












