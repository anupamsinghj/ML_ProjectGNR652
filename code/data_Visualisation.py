######## data visulaisation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
df = pd.read_csv ('final_data_all_feature.csv ')

###### printing top 5 DATASET
print(df.head())

# Information about dataset
print(df.info())
print(df.describe().T[['mean','max','min','std']])

dfa=df
plt.scatter(x='Age',y='Pregnancies',data=dfa)
plt.show()

######## their are some infeasible data, so we have to clean the data ###################
b = df[((df['Pregnancies']>=4) & (df['Age'] <= 21)) | ((df['Pregnancies']>=5) & (df['Age'] <= 22)) | ((df['Pregnancies']>=6) & (df['Age'] <= 23)) | ((df['Pregnancies']>=7) & (df['Age'] <= 24)) | ((df['Pregnancies']>=8) & (df['Age'] <= 25)) ]
a=b.index
df_new=df.drop(a)

################ comparing the dataset #######################

fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[1].scatter(x='Age',y='Pregnancies',data=df_new)
axes[1].set_title('Cleaned dataset')
plt.xlabel('Age')
plt.ylabel('Pregnancies')


axes[0].scatter(x='Age',y='Pregnancies',data=dfa)
axes[0].set_title("Old dataset")
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Pregnancies')
plt.show()

# finding the correllation
cor = dfa.corr()
print(cor['Diabetic'])

# countplot for diabetic
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Diabetic', data=df_new)
plt.show()


### lmplot between 
fig = plt.figure(figsize=(8,4), dpi=500)
sns.lmplot(x='DiastolicBloodPressure',y='Diabetic',data=dfa)
plt.show()
fig = plt.figure(figsize=(6,4))
sns.boxplot(y='DiastolicBloodPressure', x='Diabetic',data=dfa)
plt.show()




fig = plt.figure(figsize=(8,4), dpi=500)
sns.lmplot(x='PlasmaGlucose',y='Diabetic',data=dfa)
plt.show()
fig = plt.figure(figsize=(6,4))
sns.boxplot(y='PlasmaGlucose', x='Diabetic',data=dfa)
plt.show()




fig = plt.figure(figsize=(8,4), dpi=500)
sns.lmplot(x='TricepsThickness',y='Diabetic',data=dfa)
plt.show()
fig = plt.figure(figsize=(6,4))
sns.boxplot(y='TricepsThickness', x='Diabetic',data=dfa)
plt.show()



fig = plt.figure(figsize=(5,2))#, dpi=500)
sns.lmplot(x='DiabetesPedigree',y='Diabetic',data=dfa)
plt.show()
fig = plt.figure(figsize=(6,4))#, dpi=500)
sns.boxplot(y='DiabetesPedigree', x='Diabetic',data=dfa)
plt.show()
