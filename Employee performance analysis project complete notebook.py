#!/usr/bin/env python
# coding: utf-8

# ## EMPLOYEE PERFORMANCE ANALYSIS PROJECT (INX Future Inc.)

# ## INTRODUCTION
# The employee performance analysis project of INX Future Inc. is carried out with the main object of gaining actionable insights for the current employee data of the enterprise and find the causes of employee performance issues faced by the company.

# ## DATASET INFORMATION
# The employee performance date of INX Future Inc. was accessed with the following link provided by IABAC:
# http://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls
# 
# The provide dataset included 28 columns(variables) and 1200 rows(data points)
# Out of the 28 variables, 27 are predictor variables and one is the target variable
# 
# The predictor variables provided are:
# 
#        'Age','Gender', 'Education background', 'Marital status',
#        'Employee department', 'Employee job role', 'Business Travel Frequency',
#        'Distance from home', 'Employee education level', 'Employee environment satisfaction',
#        'Employee hourly rate', 'Employee job involvement', 'Employee job level',
#        'Employee job satisfaction', 'Number of companies Worked', 'OverTime',
#        'Employee last salary hike percent', 'Employee relationship satisfaction',
#        'Total work experience in years', 'Training times last year',
#        'Employee work life balance', 'Experience years at this company',
#        'Experience years in current role', 'Years since last promotion',
#        'Years With current manager' 'Attrition'
# 
# The target variable is:
#        
#        'Performance rating'
#         

# ## STEP 1: 
# ### Importing necessary libraries and packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import RandomizedSearchCV
import joblib


# ## STEP 2: 
# ### Loading the dataset

# In[2]:


pwd


# In[3]:


cd "C:\Users\arshad\Downloads"


# In[4]:


data=pd.read_excel("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls")


# ## STEP 3: 
# ### Exploratory data analysis

# In[5]:


data.head()


# In[6]:


data.tail()


# #### 3.1:Checking metadata

# In[7]:


data.info()


# In[8]:


data.describe()


# #### 3.2: Checking null values

# In[9]:


data.isnull().sum()


# #### 3.3: Analysing department wise employee performance

# In[10]:


department_wise_data=data[['EmpDepartment','PerformanceRating']]


# In[11]:


department_wise_data.head()


# In[12]:


department_wise_data['EmpDepartment'].unique()


# In[13]:


department_wise_performances=[]
for i in department_wise_data['EmpDepartment'].unique():
    print('The mean perfomance rating in',
          i,
          'department is',
          department_wise_data[department_wise_data['EmpDepartment']==i]['PerformanceRating'].mean())
    department_wise_performances.append(
        department_wise_data[department_wise_data['EmpDepartment']==i]['PerformanceRating'].mean())


# In[40]:


plt.figure(figsize=(18,6))
plt.bar(department_wise_data['EmpDepartment'].unique(),department_wise_performances,width=0.5,color='y',)
plt.xlabel('DEPARTMENTS')
plt.ylabel('Average employee performance')
for i,v in enumerate(department_wise_performances):
    plt.text(i-0.1,v+0.05,str(v))


# #### Results of Department wise employee performance anlysis:
# As obvious from the above barchart, the Development department has the highest average employee performance followed by Data science department. The rest of the deparments has more or less the same employee performance

# #### 3.4: Checking feature importances

# In[15]:


data.drop('EmpNumber',axis=1,inplace=True) # employee number is irrelevant as a predictor variable, so it can be eliminated


# In[16]:


# ENCODING CATEGORICAL VARIABLES
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()


# In[17]:


for i in data.columns:
    if data[i].dtype=='object':
        data[i]=encoder.fit_transform(data[[i]])


# In[18]:


data.info()


# In[19]:


# SPLITTING TARGET AND PREDICTOR VARIABLES
x=data.iloc[:,:26]
y=data.iloc[:,26]


# In[20]:


x.head()


# In[21]:


y.head()


# In[22]:


tester=SelectKBest(score_func=chi2,k='all')
fit=tester.fit(x,y)


# In[23]:


fit.scores_


# In[24]:


plt.figure(figsize=(18,10))
plt.barh(x.columns, fit.scores_,color='r')
for i,v in enumerate(fit.scores_):
    plt.text(v,i,str(v))


# #### 3.5: Comparing high and low performance employees
# The high performance employees(performance rating=4) and low performance employees(performance rating=2) are compared using the top 6 factors influencing the employee performance rate

# In[25]:


good_employees=data[data['PerformanceRating']==4]
bad_employees=data[data['PerformanceRating']==2]


# In[26]:


top_factors=['EmpLastSalaryHikePercent','ExperienceYearsAtThisCompany',
             'ExperienceYearsInCurrentRole',
             'YearsSinceLastPromotion','EmpEnvironmentSatisfaction',
            'YearsWithCurrManager']
a=list(range(1,7))
b=list(zip(top_factors,a))
plt.figure(figsize=(15,13))
for i,v in b:
    plt.subplot(2,3,v)
    plt.bar(['good employees','bad employees'],[np.round(good_employees[i].mean()),np.round(bad_employees[i].mean())],
            width=0.6,color='g')
    plt.ylabel(i,fontweight=1000,fontsize=15)
    for i,v in enumerate([np.round(good_employees[i].mean()),np.round(bad_employees[i].mean())]):
        plt.text(i-0.08,v,str(v),fontweight=1000,fontsize=15)
    


# #### Results of Employee comparison
# From the above bar graphs, the following conclusions can be made:
# - Higher salary hike has resulted in better employee performance
# - Employees with more experience in the company exhibit comapritively poor performance than their good performance               counterparts. This maybe due to lower satisfaction level of the experienced employees in their current role. This statement     is supported by the 'employee environment satisfaction' bar graph,indicating low level of satisfaction in poor performance     employees.
# - Years since last promotion is indirectly proportional to the employee performance.
# 
# 

# #### Recommendations made from employee comparison
# Inorder to improve the employee performance, the following points can be considered by the authorities:
# - Salary increment level can be optimized and made equal for all the employees
# - Employees can be provided with more incentives and improved working environment inorder to elevate their job satisfaction       levels
# - Frequency of promotions to employees can increased.
# 
# 

# #### 3.6: Checking multicollinearity between predictor variables

# In[27]:


plt.figure(figsize=(20,20))
correlation=x.corr()
sns.heatmap(correlation,annot=True,square=True,cmap='Blues')


# #### Results of multicollinearity analysis:
# Majority of the factors exhibit low correlation level.
# Moderate correlations(0.5 to 0.7) are observed between:
# - Employee job level & Age
# - Total work experience & Age
# - Employee job level & Experience at this company
# - Employee department & Employee job role
# - Total work experience & Experience at this company
# - Years since last promotion & Experience in current role
# - Years since last promotion & experience in this company 
# High correlations(0.7 to 0.9) are observed between:
# - Total work experience & Employee job level
# - Years with current manager & Experience at this company
# - Experience in current role & Experience at this company
# - Experience in current role & Years with current manager
# 
# Since there are no strong/very high correlations(0.9-1.0) between any of the features and the number of features are not too high to considerably increase the computational complexity, feature reduction can be avoided.

# #### 3.7: Checking the imbalance in target variable values

# In[28]:


sns.countplot(y)


# #### Results of imbalance check:
# The target variable values provided in the dataset are imbalanced and is recommended to be equalized to avoid bias in the model. 

# ## STEP 4:
# ### Data pre-processing

# #### Note:
# - Important data pre-processing steps like splitting predictor and target variables and encoding of categorical variables has     already been done as a part of the exploratory data analysis

# #### 4.1 Scaling the continuous variables

# In[29]:


for i in x.columns:
    print('the unique values of',i,'are')
    print(x[i].unique())
    print(x[i].dtype)


# In[30]:


scaler=MinMaxScaler()
scalable_variables=['YearsWithCurrManager',
                   'ExperienceYearsInCurrentRole',
                   'ExperienceYearsAtThisCompany',
                   'TotalWorkExperienceInYears',
                  'EmpLastSalaryHikePercent',
                  'EmpHourlyRate',
                  'DistanceFromHome',
                  'Age']
for i in scalable_variables:
    x[i]=scaler.fit_transform(x[[i]])


# In[31]:


x.head()


# #### 4.2: Splitting dataset into training and testing data

# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)


# #### 4.3: Adressing imbalance in data by using synthetic minority oversampling technique(SMOTE)

# In[33]:


sns.countplot(y_train)


# In[34]:


oversampler=SMOTE()
x_train,y_train=oversampler.fit_resample(x_train,y_train)


# In[35]:


sns.countplot(y_train)


# In[37]:


x.to_excel('final_data.xlsx')


# In[38]:


y.to_excel('final_data2.xlsx')
x_train.to_excel('train.xlsx')
x_test.to_excel('test.xlsx')


# In[ ]:





# ## STEP 5: 
# ### Building Macine Learning models, Evaluating their performance, and selecting the best one

# #### 5.1: Defining the available classification models

# In[36]:


logr=LogisticRegression()
knn=KNeighborsClassifier(n_neighbors=3)
forest=RandomForestClassifier(max_depth=2)
adab=AdaBoostClassifier()
xgb=XGBClassifier(max_depth=2)


# #### 5.2: Training the models

# In[37]:


logr.fit(x_train,y_train)


# In[38]:


knn.fit(x_train,y_train)


# In[39]:


forest.fit(x_train,y_train)


# In[40]:


adab.fit(x_train,y_train)


# In[41]:


xgb.fit(x_train,y_train)


# #### 5.3: Evaluating model performances

# In[42]:


model_list=[logr,knn,forest,adab,xgb]
model_accuracies=[]
for i in model_list:
    print('accuracy score for',i,'model is')
    print(accuracy_score(y_true=y_test,y_pred=i.predict(x_test.values)))
    model_accuracies.append(accuracy_score(y_true=y_test,y_pred=i.predict(x_test.values)))


# In[59]:


models=['Logistic regression','K Nearest neighbour','Random forest','Adaboost','Xgboost']
plt.figure(figsize=(8,5))
plt.barh(models,model_accuracies,color='m')
plt.xlabel('Test accuracies')
for i,v in enumerate(model_accuracies):
    plt.text(v,i,str(v),fontweight=1000)


# #### Model performance evaluation results:
# From the output obtained in last cell, it is clear that the XGBClassifier(extreme grandient boosting classification) is the most effective machine learning model for the given task.

# #### 5.4: Hyperparameter tuning of XGBClassifier

# In[54]:


parameters={'max_depth':[2,3,4,5,6],'n_estimators':[250,550,750,1000,1500,2000],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6]}
model=XGBClassifier()
hp_tuner=RandomizedSearchCV(model,parameters,cv=3)


# In[55]:


hp_tuner.fit(x,y)


# #### 5.5: Obtaining the model with best combination of parameters

# In[56]:


hp_tuner.best_params_


# In[58]:


model=hp_tuner.best_estimator_


# ## STEP 6:
# ### Testing the Best model

# In[59]:


predictions=model.predict(x_test)


# In[60]:


accuracy_score(y_true=y_test,y_pred=predictions)


# ## STEP 7:
# ### Saving the best model

# In[ ]:


joblib.dump(model,'Employee_performance_model(INX_Future_inc)')


# ## PROJECT SUMMARY
# 
# The objectives of this project included:
# 1. Department wise performances
# 2. Top 3 Important Factors effecting employee performance
# 3. A trained model which can predict the employee performance based on factors as inputs. This will be used to hire employees.
# 4. Recommendations to improve the employee performance based on insights from analysis.
# 
# Exploratory data analysis of the given dataset was carried out in first part of the project to understand the current employee data provided and to gain useful insights.
# 
# - The dataset was complete without any null values.
# - Information regarding department wise employee performances was obtained by calculating the average employee performance and   plotting a bar chart(matplotlib) of the same.
# - Feature importances were delineated by calculating the chi-squared statistic(scikit learn), which indicates the dependance of   the feature     to the target variable.
# - To understand the root cause of the performance issue, The high and low performance employees were compared with their         average values for the top 6 important features. From the insights obtained from this, appropriate recommendations were made.
# - Multicollinearity was checked by plotting a heatmap(seaborn) and no strong/ very high correlations were observed.
# - Imbalance in target variable values were determined using a countplot(seaborn) and an imbalance was present.
# 
# In the next part, data pre-processing was done to prepare the data for training and testing the machine learning models.
# 
# - Important data pre-processing steps like splitting predictor and target variables and encoding of categorical variables(using   OrdinalEncoder from scikit learn) was already done as a part of the exploratory data analysis.
# - The continuous variables were scaled using the MinMaxscaler (scikit learn) to convert the data points into values in between
#   0 and 1.
# - Data was split in to training and testing data (scikit learn).
# - Target variable imbalance in the data was addressed by using synthetic minority oversampling technique(SMOTE) (imblearn).
# 
# The next step was to choose a suitable machine learning model for the problem in hand. For this, the availabe machine learning algorithms for multi-class classification were tried out to check their accuracies in classifying the data points into 1 of the 3 performance classes.
# 
# - XGBoost algorithm poved to have the best accuracy(93.33%) and it was employed for the project.
# - Hyperparameter tuning oF XGBoost algorith was carried out using RandomizedSearchCV (scikit learn) function to obtain the model with best       parameter combination.
# - The best model gave a test accuracy of 100% and it was saved for deployment using the joblib package.
