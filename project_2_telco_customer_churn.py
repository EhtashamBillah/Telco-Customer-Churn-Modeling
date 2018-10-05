
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


plt.figure(figsize = (10,6))
sns.heatmap(df.isna(),yticklabels=False,cmap ='viridis')


# In[6]:


df[df['TotalCharges'].isna() == True]


# In[7]:


## converting 'SeniorCitizen','tenure','MonthlyCharges','TotalCharges' column from string to float
df[['tenure','MonthlyCharges','TotalCharges']] = df[['tenure','MonthlyCharges','TotalCharges']].convert_objects(convert_numeric=True)


# In[8]:


df[df['TotalCharges'].isna()==True]


# In[9]:


# filling the missing values of TotalCharges from the Monthly Charges
df2 = df.fillna(method = 'ffill',axis = 1)


# In[10]:


## converting 'SeniorCitizen','tenure','MonthlyCharges','TotalCharges' column from string to float
df2[['tenure','MonthlyCharges','TotalCharges']] = df2[['tenure','MonthlyCharges','TotalCharges']].convert_objects(convert_numeric=True)


# In[11]:


df2[df2['TotalCharges'].isna() == True]


# In[12]:


df2.iloc[488]#753,936,1082,1340


# In[13]:


df2.info()


# In[14]:


df.groupby('Churn').size()


# In[15]:


df.skew()


# # Visualization

# In[16]:


plt.figure(figsize = (10,6))
sns.heatmap(df2.corr(),cmap ='plasma',annot=True)


# In[17]:


sns.set_style('darkgrid')
sns.pairplot(data=df2,hue='Churn',palette='PuOr',
            plot_kws={'s':25,'edgecolor':'k','linewidth':0.2}, diag_kws={'edgecolor':'k','linewidth':0.6})


# In[18]:


df2.hist(figsize=(10,6),color = '#00bfff',alpha = 0.3,edgecolor = 'k',linewidth=0.3,bins=30)
plt.show()


# In[19]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Churn',y='MonthlyCharges',data = df2,palette='PuOr')


# In[20]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Churn',y='TotalCharges',data =df2)


# In[21]:


plt.figure(figsize=(10,6))
sns.boxplot(x='PhoneService',y='MonthlyCharges',data = df2,palette='Accent')


# In[22]:


plt.figure(figsize=(10,6))
sns.boxplot(x='PaymentMethod',y='TotalCharges',data = df2,palette='PiYG')


# In[23]:


df.columns


# In[24]:


plt.figure(figsize=(10,6))
sns.violinplot(x='gender',y='MonthlyCharges',data = df2,palette='Spectral')


# In[25]:


plt.figure(figsize=(10,6))
sns.kdeplot(data=df2[df2['Churn'] == 'Yes']['TotalCharges'],
            data2 = df2[df2['Churn'] == 'Yes']['MonthlyCharges'],
            cmap = 'plasma', shade=True, shade_lowest=False)
plt.show()


# In[26]:


sns.set_style('darkgrid')
plt.figure(figsize=(10,6))
plt.hist(df2[df2['Dependents'] == 'No']['MonthlyCharges'],bins=30,
        color = '#600787',edgecolor='k',linewidth=0.4,alpha = 0.7,label = 'Dependents = No')
plt.hist(df2[df2['Dependents'] == 'Yes']['MonthlyCharges'],bins=30,
        color = '#83ccd2',edgecolor='k',linewidth = 0.4,alpha = 0.7,label = 'Dependents = Yes')
plt.legend()
plt.xlabel('Monthly Charges')
plt.show()


# In[27]:


sns.set_style('darkgrid')
plt.figure(figsize=(10,6))
plt.hist(df2[df2['Contract'] == 'Month-to-month']['MonthlyCharges'],bins=30,
        color = '#cd93cc',edgecolor='k',linewidth=0.4,alpha = 0.9,label = 'Contract = Month-to-month')
plt.hist(df2[df2['Contract'] == 'Two year']['MonthlyCharges'],bins=30,
        color = '#b96554',edgecolor='k',linewidth = 0.4,alpha = 0.9,label = 'Contract = Two year')
plt.hist(df2[df2['Contract'] == 'One year']['MonthlyCharges'],bins=30,
        color = '#84b7b8',edgecolor='k',linewidth = 0.4,alpha = 0.9,label = 'Contract = One year')

plt.legend()
plt.xlabel('Monthly Charges')
plt.show()


# In[28]:


sns.lmplot(y='MonthlyCharges',x='tenure',data = df2,
           size=8,col ='gender',palette='bwr',hue = 'Churn',
          scatter_kws= {'s':22,'edgecolor':'k','linewidth':0.2})


# In[29]:


df3 = df2.drop('customerID', axis = 1)


# In[30]:


# creating dummy variables
cols = ['gender','SeniorCitizen','Partner', 'Dependents','PhoneService', 
         'MultipleLines', 'InternetService', 'OnlineSecurity',
         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',]
final_data = pd.get_dummies(data = df3,columns =cols,drop_first=True)


# In[31]:


final_data.head()


# In[32]:


from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
encoder = LabelEncoder()


# In[33]:


x= final_data.drop('Churn',axis =1)
y = final_data['Churn']
# encoding Yes/no to 1/0
y = encoder.fit_transform(y)
test_size = 0.30
seed = 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)


# In[34]:


scaler = StandardScaler()


# In[35]:


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# ** model comparison **

# In[36]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('ADA' , AdaBoostClassifier()))
models.append(('GBT' , GradientBoostingClassifier()))
models.append(('ANN' , MLPClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = " {}: {} {}".format(name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize=(10,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score


# In[38]:


lg = LogisticRegression()


# ** CROSS VALIDATION **

# In[39]:


kfold = KFold(n_splits=5,random_state=seed)
C_grid = [0.01,0.1,0.5,0.1]
intercept_grid = [True,False]
param_grid = dict(C=C_grid,fit_intercept=intercept_grid)
grid = GridSearchCV(estimator=lg,param_grid = param_grid,n_jobs=-1,cv=kfold,verbose=1,scoring='roc_auc')
grid_model = grid.fit(x_train_scaled,y_train)
y_hat_lg = grid_model.predict(x_test_scaled)
print(classification_report(y_test,y_hat_lg))
print(confusion_matrix(y_test,y_hat_lg))
roc_auc_score(y_test,y_hat_lg)


# In[40]:


print('Grid Scores:')
grid_model.grid_scores_


# In[41]:


print('Optimum Parameters ==> {}'.format(grid_model.best_params_)) 


# ** class prediction error**

# In[42]:


from yellowbrick.classifier import ClassPredictionError
classes = ['No' ,'Yes']
visualizer = ClassPredictionError(
    lg, classes=classes
)

# Fit the training data to the visualizer
visualizer.fit(x_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(x_test_scaled, y_test)

# Draw visualization
g = visualizer.poof()


# ** confusion matrix **

# *** actual class borabor sensitivity and specificity***

# In[43]:


from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(lg, classes=[0,1])

# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(x_train_scaled, y_train)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(x_test_scaled, y_test)

# How did we do?
cm.poof()


# ** classification report **

# In[44]:


from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(lg, classes=classes, support=True)

visualizer.fit(x_train_scaled, y_train)  # Fit the visualizer and the model
visualizer.score(x_test_scaled, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[45]:


print(classification_report(y_test,y_hat_lg))


# ** roc curve **

# In[46]:


from yellowbrick.classifier import ROCAUC

# Instantiate the classification model and visualizer
visualizer = ROCAUC(lg)

visualizer.fit(x_train_scaled, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test_scaled, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[47]:


from sklearn.model_selection import cross_val_score


# In[48]:


cross_val_score(lg,x_train_scaled,y_train,scoring = 'roc_auc',cv=10).mean()


# In[49]:


cross_val_score(lg,x_test_scaled,y_test,scoring = 'roc_auc',cv=10).mean()


# In[50]:


roc_auc_score(y_test,y_hat_lg)


# In[51]:


print('Grid Scores:')
grid_model.grid_scores_

