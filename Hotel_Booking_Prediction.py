import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sas

df = pd.read_csv('D:\Intern\Machine_Learning_Projects\Hotel_Booking_Prediction/hotel_bookings.csv')
df.head
df.shape
df.isna().sum()
def data_clean(df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())
data_clean(df)
df.columns
list=['adults', 'children', 'babies']
for i in list:
    print('{} has uniques values as {}'.format(i,df[i].unique()))

pd.set_option('display.max_columns',32)
filter = (df['children']==0)&(df['adults']==0)&(df['babies']==0)
df[filter]
data = df[~filter]
data.head()
country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns = ['Country','No of guests']
print(country_wise_data)

get_ipython().system('pip install folium')
import folium
from folium.plugins import HeatMap
folium.Map()
get_ipython().system('pip install plotly')
import plotly.express as px
map_guest=px.choropleth(country_wise_data,
             locations=country_wise_data['Country'],
             color=country_wise_data['No of guests'],
             hover_name=country_wise_data['Country'],
             title='Home country of Guests')
map_guest.show()
data.head()

data2=data[data['is_canceled']==0]
data2.columns
plt.figure(figsize=(12,8))
sas.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.title('Price of room type per night and per person')
plt.xlabel('Room type')
plt.ylabel('Price(Euro)')
plt.legend()
plt.show()

data_resort=data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]
data_city=data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]
data_resort.head()
resort_hotel=data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel

final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final.columns=['Month','Price_for_resort_hotel','Price_for_city_hotel']
final

get_ipython().system('pip install sorted-months-weekdays    #Installing packages  ')
get_ipython().system('pip install sort-dataframeby-monthorweek #Installing packages')
import sort_dataframeby_monthorweek as sd #importing package and giving an alias
def sort_data(df,colname):                          #function taking inputs - dataframe and column name
    return sd.Sort_Dataframeby_Month(df,colname)    #returning sorted order of month
final=sort_data(final,'Month') #Calling the function and giving parameters
final                  #printing the table
final.columns                   #Checking column names
px.line(final,x='Month',y=['Price_for_resort_hotel', 'Price_for_city_hotel'],title='Room price per night over the months')           #plotting table using line


# # Analysing Demand of Hotels
data_resort.head()         #watching dataframe
rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()     #accessing feature and converting into data frame
rush_resort.columns=['Month','No of guests']              #renaming column name
rush_resort                                       #printing the data frame
rush_city=data_city['arrival_date_month'].value_counts().reset_index()     #accessing feature and converting into data frame
rush_city.columns=['Month','No of guests']              #renaming column name
rush_city                                       #printing the data frame
final_rush=rush_resort.merge(rush_city,on='Month')     #Merging rush resort with rush city
final_rush.columns=['Month','No of guests in resort','No of guests of city hotel']   #Renaming new column name
final_rush=sort_data(final_rush,'Month')       #Sorting month column in hierarchical manner
final_rush
final_rush.columns   #Retrieving column name
px.line(final_rush,x='Month',y=['No of guests in resort', 'No of guests of city hotel'],title='Total number of guests per month')
data.head()
data.corr()             #finding correlation

co_relation=data.corr()['is_canceled']       #findint correlation with respect to is_canceled
co_relation
co_relation.abs().sort_values(ascending=False)        #used abs to avoid negative values and sort_values to sort the value of correlation
data.groupby('is_canceled')['reservation_status'].value_counts()   #Checking reservation_status on the basis of is_canceled
list_not=['days_in_waiting_list','arrival_date_year']     #excluding this feature
num_features=[col for col in data.columns if data[col].dtype!='O' and col not in list_not]     #fetching numerical colums 
num_features

data.columns             #showing all columns


# In[191]:


cat_not=['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']                #excluding categorical columns
cat_features=[col for col in data.columns if data[col].dtype=='O' and col not in cat_not]        #list comprehension
cat_features                                                                              
data_cat=data[cat_features]        #pushing cat_features to datafrmae
data_cat      #executing 
data_cat.dtypes

import warnings
from warnings import filterwarnings      #importing filterwarnings from warnings package
filterwarnings('ignore')                 #ingnoring warnings
data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])   #converting into datetime format and updating data_cat feature
data_cat.drop('reservation_status_date',axis=1,inplace=True)            #dropping reservation_status_date and updating by inplace true
data_cat['cancellation']=data['is_canceled']               #inserting column
data_cat.head()

data_cat['market_segment'].unique()      #showing unique directory of market_segment
cols = data_cat.columns                             #showing colums from 0 to 8 as we don't need cancellation column
cols
data_cat.groupby(['hotel'])['cancellation'].mean()       #accessing hotels
for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean())        #performing mean coding for each and every feature
    print('\n')                                                  #added new line to make it more user friendly
for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()        #converted into dictionary
    data_cat[col]=data_cat[col].map(dict)                                #mapping the dictionary and updating data column

data_cat.head()          #for showing few rows
dataframe=pd.concat([data_cat,data[num_features]],axis=1)     #concatenating in vertical fashion that's why axis=1
dataframe.head()      #dataframe showing
dataframe.drop('cancellation',axis=1,inplace=True)             #dropping cancellation and updating dataframe by inplace true
dataframe.shape                                                #For showing shape of dataframe
sas.distplot(dataframe['lead_time'])       #Making distribution plot of lead_time by accessing dataframe

import numpy as np      
def handle_outlier(col):            #taking log of lead_time time for greater extent of skewness 
    dataframe[col]=np.log1p(dataframe[col])


handle_outlier('lead_time')     #calling the function
sas.distplot(dataframe['lead_time'])              #showing distribution plot log applied lead_time
sas.distplot(dataframe['adr'])                      #distribution plot of adr
handle_outlier('adr')                                         #handling outlier for adr
sas.distplot(dataframe['adr'].dropna())                       #distribution plot of adr and handling missing values by dropna  
dataframe.isnull().sum()            #checking null values and doing their sum
dataframe.dropna(inplace=True)          #dropping null values and updating dataframe


y=dataframe['is_canceled']              #predicting independent features -> is_canceled
x=dataframe.drop('is_canceled',axis=1)     #dropping is_canceled feature

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel        #for selecting important features
feature_sel_model=SelectFromModel(Lasso(alpha=.005,random_state=0))                   #Specifying lasso regression model & setting low alpha value and putting random_state=0
feature_sel_model.fit(x,y)                        #fitting data to object
feature_sel_model.get_support()         #getting all the values from list
cols=x.columns          #all the columns
selected_feat=cols[feature_sel_model.get_support()]                     #adding filters to column
print('total_features {}'.format(x.shape[1]))                   #printing total features
print('selected_features {}'.format(len(selected_feat)))        #printing the selected features
selected_feat                                            #printing the entire features
x=x[selected_feat]                     #updating independent dataframe
x


from sklearn.model_selection import train_test_split         #for splitting data into train and test set
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)           #taking 25% of data for testing
from sklearn.linear_model import LogisticRegression             #importing logisticregression
logreg=LogisticRegression()                                     #calling the logisticregression class
logreg.fit(X_train,y_train)                                     #fitting training data
y_pred=logreg.predict(X_test)                                   #doing prediction on test data
y_pred                                                          #printing the prediction array


from sklearn.metrics import confusion_matrix                    #importing confusion matrix
confusion_matrix(y_test,y_pred)                                 #confusion matrix of this logistic regression model
from sklearn.metrics import accuracy_score                  #importing accuracy_score to check accuracy
accuracy_score(y_test,y_pred)                               #checking accuracy_score of y test and prediction


from sklearn.model_selection import cross_val_score           #importing cross validation
score=cross_val_score(logreg,x,y,cv=10)                          #applying cross validation for achieving more accurate score
score.mean()                                                    #achieved new score by calling mean

from sklearn.naive_bayes import GaussianNB                  #importing gussain from navie_bayes algo
from sklearn.linear_model import LogisticRegression         #importing logisticregression from linear_model 
from sklearn.neighbors import KNeighborsClassifier          #importing knn algorithm
from sklearn.ensemble import RandomForestClassifier         #importing random forest from ensemble
from sklearn.tree import DecisionTreeClassifier             #importing decision tree classifier

models=[]                                                      #this blank list is for appending all algorithms
models.append(('LogisticRegression',LogisticRegression()))     #appending logisticregression and initializing it
models.append(('Navie Bayes',GaussianNB()))                      #appending Naive Bayes and initializing it
models.append(('RandomForest',RandomForestClassifier()))         #appending random forest and initializing it
models.append(('Decision Tree',DecisionTreeClassifier()))      #appending decision tree and initializing it
models.append(('KNN',KNeighborsClassifier()))                  #appending knn and initializing it


for name,model in models:                                      #iterating over models
    print(name)                                                #printing name of model
    model.fit(X_train,y_train)                             #fitting train set of x & y into model
    predictions=model.predict(X_test)                    #predicting test set of x
    
    from sklearn.metrics import confusion_matrix                    #importing confusion matrix
    print(confusion_matrix(predictions,y_test))                  #printing confusion matrix
    print('\n')
    print(accuracy_score(predictions,y_test))                   #printing accuracy_score
    print('\n')
