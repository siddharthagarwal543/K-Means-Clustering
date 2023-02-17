import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score



#getting the data
df=pd.read_csv("/media/ubantu/New Volume/Local disk/6th Sem/Machine Learning/Exp 2/Classified Data",index_col=0)
#print(df)

#Pair Plot
sns.pairplot(df,hue='TARGET CLASS')

#standardize the variablefrom sklearn.metrics import classification_report,confusion_matrixs
scaler=StandardScaler()
df1=df.drop('TARGET CLASS',axis=1)
scaler.fit(df1)
scalar_features=scaler.transform(df1)
df_feat = pd.DataFrame(scalar_features,columns=df.columns[:-1])
print(df_feat.head())
#Train test split
x_train,x_test,y_train,y_test=train_test_split(scalar_features,df['TARGET CLASS'],test_size=0.30)


#using KNN
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)

#Predictions and Evaluations
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#choosing a k value
accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(score.mean())
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    error_rate.append(1-score.mean())

#For k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)

#For k=23
print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))