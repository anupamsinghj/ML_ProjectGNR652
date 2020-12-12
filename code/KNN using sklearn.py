########  knn ############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
sns.set()



dataset = pd.read_csv('final_data_4_feature.csv')
dataset.head()


X = dataset.drop(["Diabetic"], axis=1).values
Y = dataset.Diabetic.values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state =42)



accuracy = 0
score = []
for i in range(100):
    res = 0
    Y_pred = 0
    knn = KNeighborsClassifier(n_neighbors= i+1)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    res = confusion_matrix(Y_test, Y_pred)
    accuracy = (res[0,0]+res[1,1])/(res[0,0]+res[0,1]+res[1,0]+res[1,1])
    score.append(accuracy*100)





for i in range(100):
    print('Accuracy for {} neighbours\t: {}%'.format(i+1,round(score[i],2)))




plt.plot(score)
plt.title('Accuracy for various n values')
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy %')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9],labels=['1','2','3','4','5','6','7','8','9','10'])
plt.show()




print('\nMaximum accuracy is {}% for {} neighbours'.format(round(max(score),2),score.index(max(score))+1))
