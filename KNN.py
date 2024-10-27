from sklearn.datasets import load_breast_cancer 
# VERİ scklearn kütüphanesinden alındı 

import pandas  as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier# KNN algoritmasının kütüphanesi
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cancer=load_breast_cancer() 

df=pd.DataFrame(data=cancer.data, columns=cancer.feature_names)#sutun isimlerini belirledik
df["target"]=cancer.target # hedef sutun eklendi

X=cancer.data # features
y=cancer.target # hedef değişken 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3 , random_state=42)

#ölçeklendirme
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#knn modeli oluşuturuldu train edildi 
knn= KNeighborsClassifier(n_neighbors=3)# Model oluşturuldu komşu parametresini unutma 
knn.fit(X_train,y_train) # fit fonskiyonu verimizi samples+target kullanarak knn algoritmasını eğitir  

y_pred=knn.predict(X_test)# sonuçlarını değerlendirdik

accuracy=accuracy_score(y_test,y_pred)
print("Doğruluk:",accuracy)

conf_matrix=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf_matrix)

# Hiperparametre ayarı

accuracy_values=[]
k_values=[]

for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)# accuracy değerlerini accuracy içinde tutar 
    k_values.append(k)# k değerlerini k nın içine aktar 

#matplotlib kütüphanesi kullanılır     
plt.figure()
plt.plot(k_values,accuracy_values,marker="o",linestyle="-")
plt.title("k değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("doğruluk")
plt.xticks(k_values)
plt.grid(True)