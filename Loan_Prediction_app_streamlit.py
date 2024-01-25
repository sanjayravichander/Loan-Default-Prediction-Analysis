#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Loan Prediction
import pandas as pd
df=pd.read_csv("C:\\Users\\DELL\\Downloads\\Final Projects\\loan_default_prediction_project.csv")


# In[ ]:


## Data Preprocessing
# Step 1
#Data Cleaning:
#Handle missing data: Decide whether to impute missing values or remove instances with missing values.


# In[5]:


##Info on data
df.info()


# In[6]:


#Decription of Data
df.describe()


# In[3]:


## Handling missing data
df.isnull().sum()


# In[8]:


## filling the numnerical missing values in mean and categprical missing values in mode.
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Employment_Status']=df['Employment_Status'].fillna(df['Employment_Status'].mode()[0])

##checking again the columns having any missing values
df.isnull().sum()*100/len(df)


# In[10]:


# Check for duplicates in the entire DataFrame
duplicates = df.duplicated()

# Display rows with duplicate values
if any(duplicates):
    print("Duplicate rows:")
    print(df[duplicates])
else:
    print("No duplicate rows.")


# In[9]:


#Step 2
#Outlier detection and removal: Identify and handle outliers that can skew the model.
import matplotlib.pyplot as plt
plt.boxplot(df[['Age','Income','Credit_Score','Debt_to_Income_Ratio','Existing_Loan_Balance','Loan_Amount','Interest_Rate','Loan_Duration_Months']])
plt.show()

## we got only one outlier and it won't affect the model building that much.So we are not going to remove it


# In[17]:


## Data Exploration and Analysis
#Understand the distribution of the data through visualizations.

# Numerical Features: Histogram
#This helps to understand the range, central tendency, and spread of each variable.

numerical_columns = ['Age', 'Income', 'Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance', 'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months']

# Create subplots for numerical features
import seaborn as sn
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sn.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[21]:


# Categorical Features: Bar Plots
#It provides insights into the frequency of each category.

categorical_columns = ['Gender', 'Employment_Status', 'Location','Loan_Status']

# Creating subplots for categorical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 4, i)
    sn.countplot(x=col, data=df, palette='pastel')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[26]:


## Heat map
#This heatmap will help you visually identify the strength and direction of linear relationships between the numerical variables.
# High absolute correlation values (close to 1 or -1) may indicate a strong linear relationship
#Detecting Multicollinearity:
#Multicollinearity occurs when two or more features in a dataset are highly correlated, 
#--making it difficult for a model to separate their individual effects. 
#--By examining the correlation matrix, you can identify pairs of features with high correlation values.
#--In the context of multicollinearity, you might consider dropping one of the highly correlated features
#--to improve the model's stability and interpretability.
#High correlation between features can sometimes lead to overfitting, where a model performs well on training data but poorly on new, unseen data. 

# Numerical Features for Correlation Analysis
numerical_columns = ['Age','Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance', 'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months']

# Compute the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Plot a heatmap for better visualization
plt.figure(figsize=(10, 8))
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[29]:


## Scatter Plot
sn.relplot(x='Loan_Duration_Months',y='Loan_Amount',hue='Loan_Status',data=df)


# In[30]:


#Multivariate Analysis:To see realtionships b/w all numeric variables
##Pair Plot
sn.pairplot(df)


# In[31]:


##Target Variable Analysis:
df['Loan_Status'].value_counts()


# In[32]:


##Encoding Variables:
#In Label Encoding, each category in a categorical variable is assigned a unique integer label.
#--This is typically done in alphabetical order or based on the order of appearance.
#-- Label Encoding is suitable for ordinal data, where there is an inherent order among the categories.
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['Location']=label.fit_transform(df['Location'])


# In[35]:


## If null values greater than 20% then, we need to take mean,median or mode to fill there
## To fill na values
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Employment_Status']=df['Employment_Status'].fillna(df['Employment_Status'].mode()[0])


# In[38]:


df.head(3)


# In[39]:


df['Gender'].unique()


# In[40]:


df['Employment_Status'].unique()


# In[41]:


df['Gender']=df['Gender'].map({'Male':1,'Female':0}).astype('int')
df['Employment_Status']=df['Employment_Status'].map({'Employed':1,'Unemployed':0}).astype('int')
df['Loan_Status']=df['Loan_Status'].map({'Non-Default':0,'Default':1}).astype('int')


# In[42]:


df.head(2)


# In[45]:


## Feature Scaling -- It helps in finding the distance b/w the data.
#If not , the feature with higher value range starts dominating while calculating the distance
# ML Algorithms that require the feature scaling(distance-based) -- KNN,Neural Networks,SVM,Linear & Logistic Regression
# ML Algorithms that doesn't require the feature scaling(Non-distance-based) -- Decision Tree, Random Forest,Boosting algorithms etc.
col=['Income','Credit_Score','Existing_Loan_Balance','Loan_Amount','Interest_Rate','Loan_Duration_Months']
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
df[col]=scaled_data=st.fit_transform(df[col])


# In[46]:


df.head(2)


# In[74]:


## Model Building

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
data=df.drop('Loan_Status',axis=1)
output=df['Loan_Status']

from imblearn.over_sampling import SMOTE
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(x_train, y_train)


# In[92]:


X_resampled.shape


# In[97]:


y_resampled.shape


# In[94]:


x_train.shape


# In[95]:


x_test.shape


# In[75]:


output.head(5)


# In[170]:


## Defining the models and hyperparameter tunings
models={
        'SVM':(SVC(),{'kernel':['linear','rbf']}),
        'Logistic Regression':(LogisticRegression(),{'C':[0.1,0.008,0.002],'solver':['liblinear']}),
        'Decision Tree':(DecisionTreeClassifier(),{'max_depth':[3,2,4],
                                                  'min_samples_split':[2,3,4]}),
        'Random Forest':(RandomForestClassifier(),{'n_estimators':[50,40,100,200,250,450],'max_depth':[3,2,4],
                                                  'min_samples_split':[2,3,4]}),
        'XG Boost':(XGBClassifier(),{'n_estimators':[50,40,100,200,250,450],'max_depth':[3,2,4],
                                    'learning_rate':[0.01,0.0034,0.02]                               
                                }),
        'KNN':(KNeighborsClassifier(),{'n_neighbors':[1,2,3,4],'metric':['euclidean']}),
        'Gradient Boosting':(GradientBoostingClassifier(),{'n_estimators':[50,40,100,200,250,450],'max_depth':[3,2,4],
                                                  'min_samples_split':[2,3,4]})}


# In[171]:


best_models={}
## Iterate over the models and perform Grid Search
for model_name,(model,param_grid) in models.items():
    grid_search=GridSearchCV(model,param_grid,cv=5,scoring='accuracy')
    
    grid_search.fit(X_resampled,y_resampled)
    best_models['model_name']=grid_search.best_estimator_
    print(f"Best Parameters for {model_name}:{grid_search.best_params_}")
    print(f"Best Accuracy for {model_name}:{grid_search.best_score_}\n")


# In[199]:


ran=XGBClassifier(learning_rate= 0.02, max_depth= 4, n_estimators= 450)


# In[200]:


# Apply SMOTE to the testing set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled_test, y_resampled_test = smote.fit_resample(x_test, y_test)


# In[111]:


X_resampled_test.shape


# In[100]:


y_resampled_test.shape


# In[201]:


ran.fit(X_resampled,y_resampled)


# In[202]:


y_pred=ran.predict(X_resampled_test)


# In[203]:


accuracy = accuracy_score(y_resampled_test, y_pred)
roc_auc = roc_auc_score(y_resampled_test, y_pred)
f1 = f1_score(y_resampled_test, y_pred)


# In[205]:


print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(f"F1 Score: {f1}")
print(classification_report(y_resampled_test, y_pred))


# In[208]:


import pickle
# now you can save it to a file
file = 'C:\\Users\\DELL\\Downloads\\Model for loan prediction\\ML_Model.pkl'
with open(file, 'wb') as f:
    pickle.dump(ran,f)
    
with open(file, 'rb') as f:
    k = pickle.load(f)


# In[209]:


from PIL import Image
import streamlit as st
model = pickle.load(open('C:\\Users\\DELL\\Downloads\\Model for loan prediction\\ML_Model.pkl', 'rb'))

def run():
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('C:/Users/DELL/Downloads/Loan_img.jpg') 
    img=Image.open('C:\\Users\\DELL\\Downloads\\Logo.png')
    img=img.resize((150,150))
    st.image(img,use_column_width=False)
    st.title("Bank Loan Prediction")

    Age = st.number_input('Age',value=0)
    
    gen_display = ('Male','Female')
    gen_options=list(range(len(gen_display)))
    gen = st.selectbox("Gender",gen_options,format_func=lambda x:gen_display[x])
    
    Income = st.number_input('Income',value=0)
    
    emp_display=('Employed','Unemployed')
    emp_options=list(range(len(emp_display)))
    emp=st.selectbox("Employment Status",emp_options,format_func=lambda x:emp_display[x])
    
    loc_display=('Rural','Urban','Suburban')
    loc_options=list(range(len(loc_display)))
    loc=st.selectbox("Location",loc_options,format_func=lambda x:loc_display[x])
    
    cred_display=('Between 300-500','Above 500')
    cred_options=list(range(len(cred_display)))
    cred=st.selectbox("Credit Score",cred_options,format_func=lambda x:cred_display[x])
    
    Debt_to_Income_Ratio = st.number_input('Debt to Income Ratio',value=0)
    
    Existing_Loan_Balance = st.number_input('Existing_Loan_Balance',value=0)
    
    
    Loan_Amount = st.number_input('Loan Amount',value=0)
    
    Interest_Rate = st.number_input('Interest Rate',value=0)
    
    loan_dur=['2 months','6 months','8 months','one year','16 months']
    dur_options=range(len(loan_dur))
    dur=st.selectbox("Loan Duration",dur_options,format_func=lambda x:loan_dur[x])
    
    if st.button('Submit'):
        duration=0
        if dur==0:
            duration=60
        if dur==1:
            duration=180
        if dur==2:
            duration=240
        if dur==3:
            duration=360
        if dur==4:
            duration=480
        features = [[float(Age), float(gen), float(Income), int(emp), 
                     int(loc), float(cred), float(Debt_to_Income_Ratio),int(Existing_Loan_Balance),
                     float(Loan_Amount), float(Interest_Rate), int(duration)]]

        print(features)
        prediction=model.predict(features)
        lc=[str(i) for i in prediction]
        ans=int("".join(lc))
        if ans==0:
            import time
            with st.spinner("Please wait..."):
                time.sleep(2)

            st.error(
            "**__According to our calculations you're not eligible for the loan. Sorry:)__**")
        else:
            st.success('**__Congratulations!! you will get the loan from Bank__**')
            
run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




