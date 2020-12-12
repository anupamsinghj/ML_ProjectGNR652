##### importing package ###########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
sns.set()
############# LOADING DATASET #########
dataset = pd.read_csv('final_data_4_feature.csv')

############## EXTRACTING FEATURES AND LABEL INTO DIFFERENT ARRAY ###############
X = dataset.drop(["Diabetic"], axis=1).values
Y = dataset.Diabetic.values
#############  SPLITTING THE DATASET IN TO TRAIN AND TEST DATASET
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state =42)

############# TRAIINING ALGORITHM ###############
accuracy = np.empty((10)) # DECLRATION OF VAR
for k in range(10):
    d = np.empty((len(Y_train)))
    r = np.empty((len(Y_train)))
    Y_Pred = np.empty((len(Y_test)))
    for i in range(len(Y_test)):
        for j in range(len(Y_train)):
            d[j] =np.sqrt(((X_test[i,0]-X_train[j,0])**2) +((X_test[i,1]-X_train[j,1])**2) +((X_test[i,2]-X_train[j,2])**2) +((X_test[i,3]-X_train[j,3])**2))
        r = d.copy()
        r.sort()
        ind = []
        for l in range(k+1):
            ind.append(np.asarray(np.where(d==r[l]))[0][0])
        suma=0
        for m in range(k+1):
            suma = suma + Y_train[ind[m]]
        if ((k+1)%2==0):
            if (suma >= (k+1)/2):
                Y_Pred[i] = 1
            else:
                Y_Pred[i] = 0
        else:
            if (suma > (k+1)/2):
                Y_Pred[i] = 1
            else:
                Y_Pred[i] = 0
    con_mat = confusion_matrix(Y_test,Y_Pred)
    accuracy[k] = (con_mat[0,0] + con_mat[1,1])/(con_mat[0,0] + con_mat[1,1] + con_mat[1,0] + con_mat[0,1])
    print('Accuracy for {} neighbours\t: {}%'.format(k+1,round(accuracy[k]*100,2)))

##### PLOT OF ACCURACY VS K #################
plt.plot(accuracy*100)
plt.title('Accuracy for various n values')
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy %')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9],labels=['1','2','3','4','5','6','7','8','9','10'])
plt.show()



print('Maximum accuracy is {}% for {} neighbours'.format(round(max(accuracy*100),2),np.asarray(np.where(accuracy==max(accuracy)))[0][0]+1))
