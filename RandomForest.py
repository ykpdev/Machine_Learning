from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Makine öğreniminde ensemble, birden fazla makine öğrenimi modelinin 
#bir araya getirilerek daha güçlü ve daha doğru tahminler yapmasını sağlayan bir yöntemdir.
from sklearn.metrics import accuracy_score
faces=fetch_olivetti_faces()

plt.figure()
for i in range(2): # 2 görüntüyü görselleştirmek için 
    plt.subplot(1,2,i+1)
    plt.imshow(faces.images[i],cmap="gray" )
plt.show()    

X=faces.data
y=faces.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rf_clf=RandomForestClassifier(n_estimators=100,random_state=42)#estimators ağaç sayısı 100 ağaç olacak diyor 
rf_clf.fit(X_train,y_train)

y_pred=rf_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred) # hangi ikisi arasında karşılaştırma yapmak istiyor isek onlar yazılır 

print(accuracy)