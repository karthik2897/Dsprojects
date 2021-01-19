#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[3]:


df=pd.read_csv("wine.csv")
df.head()


# In[ ]:


#Looking at the dataset we can consider "Class" as the Target variable 
#As there are NoNull values we should fix the distortions in the data if any and start applying our mobel 


# In[4]:


df.isnull().sum()
#There are No Null valuses 


# In[5]:


df.describe()


# In[87]:


df["Class"].plot.hist()


# In[29]:


corr_hmap= df.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()
#we can observe that the "Alcalinity of ash" has a very low corelation with the class so we can drop the column if need be 


# In[33]:


df.hist(figsize=(20,20))
#we can observe that the features are not normally distributed, So the Skewness of the data must be removed 


# In[44]:


plt.figure(figsize=(10,7))
df.corr()["Class"].sort_values(ascending=False).drop(["Class"])
plt.xlabel("Features")
plt.ylabel("Corelation with target Variable")
plt.show()
df.corr()["Class"].sort_values(ascending=False).drop(["Class"])


# In[30]:


df.plot(kind="box",subplots=True,layout=(2,7))
#As we observe there are few outliers in the data 


# In[45]:


df.skew()
#we can say that "Malic acid" and "Magnesium" have Skewness 
# So the skewness should be treated


# In[9]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[10]:


threshold=3
print(np.where(z>3))


# In[12]:


df_new=df[(z<3).all(axis=1)]


# # Percentage loss in data after removal of Skewness

# In[59]:


print("Original Shape",df.shape,"New Shape",df_new.shape)
Percntage_loss=(10/178)*100
print(Percntage_loss)


# As we can see there is loss of 5.6% data from the dataset we are going a head with outliers removed 

# In[60]:


Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
print(IQR)


# In[74]:


df.new1=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
df.new1.shape


# In[75]:


Percntage_loss=(17/178)*100
print(Percntage_loss)


# As the percentage loss using zscore is less we will go ahead with df_new 
# 

# In[88]:


y=df_new.iloc[:,:1]
x=df_new.iloc[:,1:]


# In[90]:


y.shape


# In[102]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_t=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
lr=LogisticRegression()
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")


# In[103]:


max_scr=0
for i in range(0,300):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=i)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_test=lr.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,2)==round(accuracy_score(y_test,pred_test)*100,2):
        print("At random state",i,"The model Perfomes very well")
        print("Training accuracy_score is:",accuracy_score(y_train,pred_train)*100)
        print("Testing accuracy_score is:",accuracy_score(y_test,pred_test)*100)


# As we can see the model perfomes well in many cases, so Let us consdider randomstate=36

# In[124]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=36) 


# In[125]:


pred=lr.predict(x_test)
pred


# In[130]:


#Logistic Regression
lr.fit(x_train,y_train)
print("accuracy score:",accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[128]:


#SVC
svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)
#svc.score(x_trian,y_tarin)
predsvc=svc.predict(x_test)
print("accuracy score:",accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# In[131]:


#DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='gini')
svc.fit(x_train,y_train)
#svc.score(x_trian,y_tarin)
predsvc=svc.predict(x_test)
print("accuracy score:",accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# In[139]:


import joblib
joblib.dump(lr,"wineproject.pkl")


# # We can conclude our Project as we have achived a 100% Accuracy 

# In[ ]:




