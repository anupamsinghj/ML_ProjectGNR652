###### DECISION TREE ########

## IMPORT LIBRARY
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#### LOADING DATASET
df=pd.read_csv('final_data_all_feature.csv')


data=df[df.Pregnancies<=4]
#newdata with all features
newdata_alf=data.drop(["PatientID"], axis=1)
#newdata with selected features
newdata = data.drop(["PatientID","PlasmaGlucose","DiastolicBloodPressure","TricepsThickness","DiabetesPedigree"], axis=1)





from sklearn import model_selection
# for all features, taking Y_alf as target variable and X_alf as input features.
Y_alf = np.asarray(newdata_alf['Diabetic'])
X_alf = np.asarray(newdata_alf.drop('Diabetic',1))
trainData_alf,testData_alf,trainLabel_alf,testLabel_alf= model_selection.train_test_split(X_alf,Y_alf,test_size=0.3,random_state=1)

# for selected featurs
Y = np.asarray(newdata['Diabetic'])
X = np.asarray(newdata.drop('Diabetic',1))
trainData,testData,trainLabel,testLabel= model_selection.train_test_split(X,Y,test_size=0.3,random_state=1)




#### FITTING MODEL
from sklearn import tree
dectr_alf = tree.DecisionTreeClassifier()
dectr_alf.fit(trainData_alf,trainLabel_alf)

dectr = tree.DecisionTreeClassifier()
dectr.fit(trainData,trainLabel)



## PREDICTING
pred_Label_alf=dectr_alf.predict(testData_alf)
pred_Label=dectr.predict(testData)


from sklearn.metrics import accuracy_score
print("MEASURES OF RANDOMNESS IS : 'GAIN' ")
print("ACUURACY FOR ALL FEATURE : ")
print (round(accuracy_score(testLabel_alf,pred_Label_alf )*100,2))
print("ACUURACY FOR SELECTED FEATURE : ")
print (round(accuracy_score(testLabel,pred_Label )*100,2))



# In[114]:


from sklearn.metrics import confusion_matrix
print("confusion_matrix FOR all FEATURES")
print(confusion_matrix(testLabel_alf,pred_Label_alf))
print("confusion_matrix FOR SELECTED FEATURES")
print(confusion_matrix(testLabel,pred_Label))




########## FITTING MODEL FOR "ENTROPY" AS A MEASURE FOR SIMLARITY/RANDOMNESS
print("\n PREDICTION FOR TAKING 'ENTROPY' TO MEASURE NOISE/RANDOMNESS\n" )
dectr_alf = tree.DecisionTreeClassifier(criterion='entropy')
dectr_alf.fit(trainData_alf,trainLabel_alf)

dectr = tree.DecisionTreeClassifier(criterion='entropy')
dectr.fit(trainData,trainLabel)



pred_Label_alf=dectr_alf.predict(testData_alf)
pred_Label=dectr.predict(testData)



print("ACUURACY FOR ALL FEATURE : ")
print (round(accuracy_score(testLabel_alf,pred_Label_alf )*100,2))
print("ACUURACY FOR SELECTED FEATURE : ")
print (round(accuracy_score(testLabel,pred_Label )*100,2))

print("confusion_matrix FOR all FEATURES")
print(confusion_matrix(testLabel_alf,pred_Label_alf))
print("confusion_matrix FOR SELECTED FEATURES")
print(confusion_matrix(testLabel,pred_Label))
