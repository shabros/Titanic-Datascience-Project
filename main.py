#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Science Solutions
# 
# 
# The notebook walks us through a typical workflow for solving data science competitions at sites like Kaggle.
# 
# # Question
# 
# - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# - One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# 

# In[119]:


#importing libraries
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
#**********************************************************
#Acquiring dataset
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
together=[train_data, test_data]
#**************************************************************
#describing data
print(train_data.columns.values)
train_data.head()
train_data.drop(columns="Survived")
#****************************************************************
#Checking for missing data
print(train_data.isnull().sum())
print('_'*40)
print(test_data.isnull().sum())
#************************************************************


# In[120]:


#describing data
print(train_data.columns.values)
train_data.head()
train_data.drop(columns="Survived")
#*********************************************************
#Checking for missing data
print(train_data.isnull().sum())
print('_'*40)
print(test_data.isnull().sum())


# In[121]:


train_data.head(30)
#we have categorical and numerical features in this dataset
#categorical: Survived, Sex, Embarked, Ordinal: Pclass
#Continous: Age, Fare. Discrete: : SibSp, Parch

#Ticket feature is an alphanumeric feature


# In[122]:


train_data.info()
print('__'*50)
test_data.info()


# In[123]:


train_data.describe()

#the total number of passenger is 891 person. 
print("total number of passenger is: " + str(train_data["PassengerId"].count()))

ax=train_data.hist(column= "Age", bins=10)
plt.title("Histomgram of age")
plt.xlabel("Age")
plt.ylabel("number of people")
#Percentage of Survived persons

print("Percentage of survived passengers out of all the passengers is: "+ str((train_data["Survived"]== 1).sum()*100/train_data["PassengerId"].count()))
print("Percentage of passengers who travel without siblings or spouse :"+ str((train_data["SibSp"]== 0).sum()*100/train_data["PassengerId"].count()))
print("Percentage of passengers who travel with one sibling or spouse :"+ str((train_data["SibSp"]== 1).sum()*100/train_data["PassengerId"].count()))      
print("Percentage of passengers who paid at most $222: " + str((train_data["Fare"] <= 222).sum()*100/train_data["PassengerId"].count())) 
print(" Percentage of passengers who were 40 or more than 40 years old: "+ str((train_data["Age"] >= 40).sum()*100/train_data["PassengerId"].count())) 
print("Percentage of survived passengers who were more than 70 years old: "+str ((train_data["Age"] > 70).sum()*100/(train_data["Survived"]==1).sum()))

print("-------rate of survived passengers in each class-------")
print("Pclass=1: "+str(((train_data["Pclass"] == 1).sum()*100)/(train_data["Survived"]==1).sum()))
print("Pclass=2: " +str((train_data["Pclass"] == 2).sum()*100/(train_data["Survived"]==1).sum()))
print("Pclass=3: "+ str((train_data["Pclass"] == 3).sum()*100/(train_data["Survived"]==1).sum()))


ax7=train_data.hist(column= "Fare", bins=10, weights=np.ones_like(train_data["Fare"]) * 100. / len(train_data["Fare"]))
# train_data["Fare"].plot(kind='hist', density=1, bins=20, stacked=False, alpha=.5)
# ax2=train_data["Fare"].hist(bins=20, weights=np.ones_like(train_data["Fare"]) * 100. / len(train_data["Fare"]))
plt.title("Histogram of Fare")
plt.xlabel("Fare")
plt.ylabel("Percentage of passenger")

print("________________rate of survived passenger_________________________")
print("the percentage of male passenger is: "+str((train_data["Sex"]== "male").sum()*100/(train_data["PassengerId"]).count()))
print("the percentage of female passenger is :"+ str((train_data["Sex"]== "female").sum()*100/(train_data["PassengerId"]).count()))
print("the percentage of male passengers who survived is: "+ str((train_data["Sex"]== "male").sum()*100/(train_data["Survived"]==1).sum()))
print("the percentage of female passengers who survived is: "+str((train_data["Sex"]== "female").sum()*100/(train_data["Survived"]==1).sum()))


# In[124]:


train_data.describe(include=['O'])


# In[125]:



print(type(train_data.groupby('Sex').sum()))
table1=train_data.groupby("Sex").sum()
print(table1)
table2=table1.drop("PassengerId",1)
table2.groupby('Sex').sum().plot(kind='bar')


# In[126]:


train_data[["Pclass", "PassengerId"]].groupby(["Pclass"], as_index=False).count()


# In[127]:


df=train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).sum() 
fig,ax=plt.bar(x=df["Sex"], height=df["Survived"])
# print(df)
print(train_data["Sex"].value_counts())


# In[128]:


da=train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)
print(da)


# In[129]:


train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[130]:



print((train_data["Age"] <= 20).sum()/(train_data["Survived"]==0).sum())
print((train_data["Age"] <= 20).sum()/(train_data["Survived"]==1).sum())

g=sns.FacetGrid(train_data, col= "Survived")
g.map(plt.hist,"Age", bins=5)


# In[131]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
#grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)


g1=sns.FacetGrid(train_data, col="Survived", row="Pclass")
g1.map(plt.hist, "Age", alpha=0.5, bins=20)
g1.add_legend();


# In[132]:



grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[133]:


#correcting
#Ticket, cabin and passengerId can be dropped from the data as they have duplication, incompleteness and no contribution to the survival rate respectively.

print("Before", train_data.shape, test_data.shape, together[0].shape, together[1].shape)
#now we drop 
train_datan = train_data.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_datan = test_data.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
togethern = [train_datan, test_datan]

"After", train_datan.shape, test_datan.shape, togethern[0].shape, togethern[1].shape


# ### Creating new feature extracting from existing
# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
# 
# **Observations.**
# 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decision.**
# 
# - We decide to retain the new Title feature for model training.

# ### Converting a categorical feature
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[134]:


train_datan.head()
train_datan["Sex"].isnull().sum()


# In[149]:



guess_ages=np.zeros((2,3))

for dataset in togethern:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# In[150]:


train_datan.head(50)


# In[137]:


train_datan = train_datan.drop(['Parch', 'SibSp'], axis=1)
test_datan = test_datan.drop(['Parch', 'SibSp'], axis=1)
togethern = [train_datan, test_datan]


# We can also create an artificial feature combining Pclass and Age.

# In[138]:


freq_data = train_datan.Embarked.mode()[0]
freq_data
for dataset in togethern:
    dataset["Embarked"]=dataset["Embarked"].fillna(freq_data)
train_datan[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[141]:


for dataset in togethern:
     dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)




# In[142]:


togethern[0].head()


# In[144]:


for dataset in togethern:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[145]:


togethern[1].head()


# In[146]:


test_datan.head(10)


# In[147]:


freq_data = train_datan.Fare.mode()[0]
freq_data
for dataset in togethern:
    dataset["Fare"]=dataset["Fare"].fillna(freq_data)


# In[151]:


test_datan.isnull().sum()


# ## Model, predict and solve
# 
# 

# In[ ]:


X_train = train_datan.drop(["Survived","Name"], axis=1)
Y_train = train_datan["Survived"]
X_test  = test_datan.drop(["Name"], axis=1)
X_train.shape, Y_train.shape, X_test.shape
X_test


# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# In[ ]:


X_test.isnull().sum()


# In[ ]:


# Logistic Regression

#reg = LogisticRegression()
#reg.fit(X_train, Y_train)
#Y_pred = reg.predict(X_test)
#acc-log = (reg.score(X_train, Y_train) * 100, 2)
#acc-log


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - Inversely as Pclass increases, probability of Survived=1 decreases the most.
# - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# - So is Title as second highest positive correlation.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knnscore=round(knn.score(X_train, Y_train) * 100, 2)
knnscore


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

