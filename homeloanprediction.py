#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv("train (1).csv")
test = pd.read_csv("test.csv")


# In[4]:


train_original = train.copy()
test_original = test.copy()


# In[5]:


train.columns


# In[6]:


test.columns


# In[7]:


train.dtypes


# In[8]:


print('Training data shape: ', train.shape)
train.head()


# In[9]:


print('Test data shape: ', test.shape)
test.head()


# In[10]:


train["Loan_Status"].count()


# In[11]:


train["Loan_Status"].value_counts()


# In[12]:


train["Loan_Status"].value_counts(normalize=True)*100


# In[13]:


train["Loan_Status"].value_counts(normalize=True).plot.bar(title = 'Loan_Status')


# In[14]:


train["Gender"].count()


# In[15]:


train["Gender"].value_counts()


# In[16]:


train['Gender'].value_counts(normalize=True)*100


# In[17]:


train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')


# In[18]:


train["Married"].count()


# In[19]:


train["Married"].value_counts()


# In[20]:


train['Married'].value_counts(normalize=True)*100


# In[21]:


train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')


# In[22]:


train["Self_Employed"].count()


# In[23]:


train["Self_Employed"].value_counts()


# In[24]:


train['Self_Employed'].value_counts(normalize=True)*100


# In[25]:


train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')


# In[26]:


train["Credit_History"].count()


# In[27]:


train["Credit_History"].value_counts()


# In[28]:


train['Credit_History'].value_counts(normalize=True)*100


# In[29]:


train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')


# In[30]:


train['Dependents'].count()


# In[31]:


train["Dependents"].value_counts()


# In[32]:


train['Dependents'].value_counts(normalize=True)*100


# In[33]:


train['Dependents'].value_counts(normalize=True).plot.bar(title="Dependents")


# In[34]:


train["Education"].count()


# In[35]:


train["Education"].value_counts()


# In[36]:


train["Education"].value_counts(normalize=True)*100


# In[37]:


train["Education"].value_counts(normalize=True).plot.bar(title = "Education")


# In[38]:


train["Property_Area"].count()


# In[39]:


train["Property_Area"].value_counts()


# In[40]:


train["Property_Area"].value_counts(normalize=True)*100


# In[41]:


train["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")


# In[42]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train["ApplicantIncome"]);

plt.subplot(122)
train["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()


# In[43]:


train.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()


# In[44]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train["CoapplicantIncome"]);

plt.subplot(122)
train["CoapplicantIncome"].plot.box(figsize=(16,5))
plt.show()


# In[45]:


plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()


# In[46]:


plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df["Loan_Amount_Term"]);

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16,5))
plt.show()


# In[47]:


print(pd.crosstab(train["Gender"],train["Loan_Status"]))
Gender = pd.crosstab(train["Gender"],train["Loan_Status"])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()


# In[48]:


print(pd.crosstab(train["Married"],train["Loan_Status"]))
Married=pd.crosstab(train["Married"],train["Loan_Status"])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Married")
plt.ylabel("Percentage")
plt.show()


# In[49]:


print(pd.crosstab(train['Dependents'],train["Loan_Status"]))
Dependents = pd.crosstab(train['Dependents'],train["Loan_Status"])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.show()


# In[50]:


print(pd.crosstab(train["Education"],train["Loan_Status"]))
Education = pd.crosstab(train["Education"],train["Loan_Status"])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.show()


# In[52]:


print(pd.crosstab(train["Self_Employed"],train["Loan_Status"]))
SelfEmployed = pd.crosstab(train["Self_Employed"],train["Loan_Status"])
SelfEmployed.div(SelfEmployed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Self_Employed")
plt.ylabel("Percentage")
plt.show()


# In[53]:


print(pd.crosstab(train["Credit_History"],train["Loan_Status"]))
CreditHistory = pd.crosstab(train["Credit_History"],train["Loan_Status"])
CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Credit_History")
plt.ylabel("Percentage")
plt.show()


# In[54]:


print(pd.crosstab(train["Property_Area"],train["Loan_Status"]))
PropertyArea = pd.crosstab(train["Property_Area"],train["Loan_Status"])
PropertyArea.div(PropertyArea.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Property_Area")
plt.ylabel("Loan_Status")
plt.show()


# In[55]:


train.groupby("Loan_Status")['ApplicantIncome'].mean().plot.bar()


# In[56]:


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)


# In[57]:


print(pd.crosstab(train["Income_bin"],train["Loan_Status"]))
Income_bin = pd.crosstab(train["Income_bin"],train["Loan_Status"])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("ApplicantIncome")
plt.ylabel("Percentage")
plt.show()


# In[58]:


bins=[0,1000,3000,42000]
group =['Low','Average','High']
train['CoapplicantIncome_bin']=pd.cut(df["CoapplicantIncome"],bins,labels=group)


# In[59]:


print(pd.crosstab(train["CoapplicantIncome_bin"],train["Loan_Status"]))
CoapplicantIncome_Bin = pd.crosstab(train["CoapplicantIncome_bin"],train["Loan_Status"])
CoapplicantIncome_Bin.div(CoapplicantIncome_Bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("CoapplicantIncome")
plt.ylabel("Percentage")
plt.show()


# In[60]:


train["TotalIncome"]=train["ApplicantIncome"]+train["CoapplicantIncome"]


# In[61]:


bins =[0,2500,4000,6000,81000]
group=['Low','Average','High','Very High']
train["TotalIncome_bin"]=pd.cut(train["TotalIncome"],bins,labels=group)


# In[62]:


print(pd.crosstab(train["TotalIncome_bin"],train["Loan_Status"]))
TotalIncome = pd.crosstab(train["TotalIncome_bin"],train["Loan_Status"])
TotalIncome.div(TotalIncome.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(2,2))
plt.xlabel("TotalIncome")
plt.ylabel("Percentage")
plt.show()


# In[63]:


bins = [0,100,200,700]
group=['Low','Average','High']
train["LoanAmount_bin"]=pd.cut(df["LoanAmount"],bins,labels=group)


# In[64]:


print(pd.crosstab(train["LoanAmount_bin"],train["Loan_Status"]))
LoanAmount=pd.crosstab(train["LoanAmount_bin"],train["Loan_Status"])
LoanAmount.div(LoanAmount.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("LoanAmount")
plt.ylabel("Percentage")
plt.show()


# In[65]:


train=train.drop(["Income_bin","CoapplicantIncome_bin","LoanAmount_bin","TotalIncome","TotalIncome_bin"],axis=1)


# In[66]:


train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)


# In[67]:


matrix = train.corr()
f, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True);


# In[68]:


train.isnull().sum()


# In[69]:


train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
train["Married"].fillna(train["Married"].mode()[0],inplace=True)
train['Dependents'].fillna(train["Dependents"].mode()[0],inplace=True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mode()[0],inplace=True)


# In[70]:


train["Loan_Amount_Term"].value_counts()


# In[71]:


train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0],inplace=True)


# In[72]:


train["Loan_Amount_Term"].value_counts()


# In[73]:


train["LoanAmount"].fillna(train["LoanAmount"].median(),inplace=True)


# In[74]:


train.isnull().sum()


# In[75]:


test.isnull().sum()


# In[76]:


test["Gender"].fillna(test["Gender"].mode()[0],inplace=True)
test['Dependents'].fillna(test["Dependents"].mode()[0],inplace=True)
test["Self_Employed"].fillna(test["Self_Employed"].mode()[0],inplace=True)
test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].mode()[0],inplace=True)
test["Credit_History"].fillna(test["Credit_History"].mode()[0],inplace=True)
test["LoanAmount"].fillna(test["LoanAmount"].median(),inplace=True)


# In[77]:


test.isnull().sum()


# In[78]:


sns.distplot(train["LoanAmount"]);


# In[79]:


train['LoanAmount'].hist(bins=20)


# In[80]:


train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)


# In[81]:


sns.distplot(train["LoanAmount_log"])


# In[82]:


test["LoanAmount_log"]=np.log(train["LoanAmount"])
test['LoanAmount_log'].hist(bins=20)


# In[83]:


sns.distplot(test["LoanAmount_log"])


# In[84]:


train["TotalIncome"]=train["ApplicantIncome"]+train["CoapplicantIncome"]


# In[85]:


train[["TotalIncome"]].head()


# In[86]:


test["TotalIncome"]=test["ApplicantIncome"]+test["CoapplicantIncome"]


# In[87]:


test[["TotalIncome"]].head()


# In[88]:


sns.distplot(train["TotalIncome"])


# In[89]:


train["TotalIncome_log"]=np.log(train["TotalIncome"])
sns.distplot(train["TotalIncome_log"])


# In[90]:


sns.distplot(test["TotalIncome"])


# In[91]:


test["TotalIncome_log"] = np.log(train["TotalIncome"])
sns.distplot(test["TotalIncome_log"])


# In[92]:


train["EMI"]=train["LoanAmount"]/train["Loan_Amount_Term"]
test["EMI"]=test["LoanAmount"]/test["Loan_Amount_Term"]


# In[93]:


train[["EMI"]].head()


# In[94]:


test[["EMI"]].head()


# In[95]:


sns.distplot(train["EMI"])


# In[96]:


sns.distplot(test["EMI"])


# In[97]:


train["Balance_Income"] = train["TotalIncome"]-train["EMI"]*1000
test["Balance_Income"] = test["TotalIncome"]-test["EMI"]


# In[98]:


train[["Balance_Income"]].head()


# In[99]:


test[["Balance_Income"]].head()


# In[100]:


train=train.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)


# In[101]:


train.head()


# In[102]:


test = test.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)


# In[103]:


test.head()


# In[104]:


train=train.drop("Loan_ID",axis=1)
test=test.drop("Loan_ID",axis=1)


# In[105]:


train.head(3)


# In[106]:


test.head(3)


# In[107]:


X=train.drop("Loan_Status",1)


# In[108]:


X.head(2)


# In[109]:


y=train[["Loan_Status"]]


# In[110]:


y.head(2)


# In[111]:


X = pd.get_dummies(X)


# In[112]:


X.head(3)


# In[113]:


train=pd.get_dummies(train)
test=pd.get_dummies(test)


# In[114]:


train.head(3)


# In[115]:


test.head(3)


# In[116]:


from sklearn.model_selection import train_test_split


# In[117]:


x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3,random_state=1)


# In[118]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[119]:


logistic_model = LogisticRegression(random_state=1)


# In[120]:


logistic_model.fit(x_train,y_train)


# In[121]:


pred_cv_logistic=logistic_model.predict(x_cv)


# In[122]:


score_logistic =accuracy_score(pred_cv_logistic,y_cv)*100


# In[123]:


score_logistic


# In[124]:


pred_test_logistic = logistic_model.predict(test)


# In[125]:


from sklearn.tree import DecisionTreeClassifier


# In[126]:


tree_model = DecisionTreeClassifier(random_state=1)


# In[127]:


tree_model.fit(x_train,y_train)


# In[128]:


pred_cv_tree=tree_model.predict(x_cv)


# In[129]:


score_tree =accuracy_score(pred_cv_tree,y_cv)*100 


# In[130]:


score_tree


# In[131]:


pred_test_tree = tree_model.predict(test)


# In[132]:


from sklearn.ensemble import RandomForestClassifier


# In[133]:


forest_model = RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)


# In[134]:


forest_model.fit(x_train,y_train)


# In[135]:


pred_cv_forest=forest_model.predict(x_cv)


# In[136]:


score_forest = accuracy_score(pred_cv_forest,y_cv)*100


# In[137]:


score_forest


# In[138]:


pred_test_forest=forest_model.predict(test)


# In[139]:


from sklearn.model_selection import GridSearchCV


# In[140]:


paramgrid = {'max_depth': list(range(1,20,2)),'n_estimators':list(range(1,200,20))}


# In[141]:


grid_search = GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)


# In[142]:


grid_search.fit(x_train,y_train)


# In[143]:


grid_search.best_estimator_


# In[144]:


grid_forest_model = RandomForestClassifier(random_state=1,max_depth=3,n_estimators=101)


# In[145]:


grid_forest_model.fit(x_train,y_train)


# In[146]:


pred_grid_forest = grid_forest_model.predict(x_cv)


# In[147]:


score_grid_forest = accuracy_score(pred_grid_forest,y_cv)*100


# In[148]:


score_grid_forest


# In[149]:


pred_grid_forest_test = grid_forest_model.predict(test)


# In[152]:


importances = pd.Series(forest_model.feature_importances_,index=X.columns)
importances.plot(kind='barh', figsize=(12,8))


# In[ ]:




