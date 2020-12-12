# Logistics models ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##### loading dataset
diabetes = pd.read_csv("final_data_4_feature.csv")
diabetes = pd.DataFrame(diabetes)
plt.hist(diabetes[diabetes.Diabetic == 1].BMI, bins=[15,25,35,45,60], alpha = 0.3)
#plt.show()

# splitting the data set in to test and training data
X = pd.DataFrame({'Pregnancies':diabetes.Pregnancies, 'SerumInsulin':diabetes.SerumInsulin, 'BMI':diabetes.BMI, 'Age':diabetes.Age})
y = diabetes.Diabetic
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# creating sigmoid function
def sigmoid(X_train,B):
    Z = np.dot(X_train,B)
    return 1/(1+np.exp(-Z))

# logistic regression with maximun likelihood with ridge regression
def logregression(X_train,y_train,B,learningrate,iterations,Lembda):
    for i in range(iterations):
        Z = sigmoid(X_train,B)
        B = B + learningrate*((np.dot((y_train - Z),X_train)) + Lembda*B)
    return B;

# finding coefficient
B = np.zeros(X_train.shape[1])
learningrate = 0.5
iterations = 50000
A = logregression(X_train,y_train,B,learningrate,iterations,.0001)
print("coefficients := " + str(A))


# finding accuracy of the model in training and testing set
y_pre = np.dot(X_train,A)
for i in range(len(y_pre)):
    if y_pre[i] >= 0.5:
        y_pre[i] = 1
    else:
        y_pre[i] = 0

print("training accuracy using hardcoding:= " + str((len(y_train)-np.sum(abs(y_pre - y_train)))/len(y_train)))

#finding the accureacy of the testing set with classification metrix
Y_pt = np.dot(X_test,A)
for i in range(len(Y_pt)):
    if Y_pt[i] >= 0.5:
        Y_pt[i] = 1
    else:
        Y_pt[i] = 0

print("testing accuracy := " + str((len(y_test)-np.sum(abs(Y_pt - y_test)))/len(y_test)))
from sklearn.metrics import classification_report
print("classification_report  for hardcoding")
print(classification_report(y_test, Y_pt, target_names=["non-Diabetic", "Diabetic"]))

#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# finding the accureacy of the model with ROC curve
from sklearn.metrics import roc_curve,roc_auc_score
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, Y_pt)
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, Y_pt))
plt.title('ROC - Logistic regression')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# FITTING decision tree
print("Acuuracy for decision tree : ")

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("classification_report  for hardcoding")
print(classification_report(y_test, y_pred, target_names=["non-Diabetic", "Diabetic"]))
