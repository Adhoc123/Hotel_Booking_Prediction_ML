#!/usr/bin/env python
# coding: utf-8

# In[139]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sas


# In[140]:


df = pd.read_csv('D:\Intern\Machine_Learning_Projects\Hotel_Booking_Prediction/hotel_bookings.csv')


# In[141]:


df.head()


# In[142]:


df.shape


# In[143]:


df.isna().sum()


# In[144]:


def data_clean(df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())


# In[145]:


data_clean(df)


# In[146]:


df.columns


# In[147]:


list=['adults', 'children', 'babies']
for i in list:
    print('{} has uniques values as {}'.format(i,df[i].unique()))


# In[148]:


pd.set_option('display.max_columns',32)


# In[149]:


filter = (df['children']==0)&(df['adults']==0)&(df['babies']==0)
df[filter]


# In[150]:


data = df[~filter]
data.head()


# In[151]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()


# In[152]:


country_wise_data.columns = ['Country','No of guests']
print(country_wise_data)


# In[153]:


get_ipython().system('pip install folium')


# In[154]:


import folium
from folium.plugins import HeatMap


# In[155]:


folium.Map()


# In[156]:


get_ipython().system('pip install plotly')


# In[157]:


import plotly.express as px


# In[158]:


map_guest=px.choropleth(country_wise_data,
             locations=country_wise_data['Country'],
             color=country_wise_data['No of guests'],
             hover_name=country_wise_data['Country'],
             title='Home country of Guests')
map_guest.show()


# In[159]:


data.head()


# In[160]:


data2=data[data['is_canceled']==0]


# In[161]:


data2.columns


# In[162]:


plt.figure(figsize=(12,8))
sas.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.title('Price of room type per night and per person')
plt.xlabel('Room type')
plt.ylabel('Price(Euro)')
plt.legend()
plt.show()


# In[163]:


data_resort=data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]
data_city=data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]


# In[164]:


data_resort.head()


# In[165]:


resort_hotel=data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel


# In[166]:


city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[167]:


final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final.columns=['Month','Price_for_resort_hotel','Price_for_city_hotel']
final


# In[168]:


get_ipython().system('pip install sorted-months-weekdays    #Installing packages  ')


# In[169]:


get_ipython().system('pip install sort-dataframeby-monthorweek #Installing packages')


# In[170]:


import sort_dataframeby_monthorweek as sd #importing package and giving an alias


# In[171]:


def sort_data(df,colname):                          #function taking inputs - dataframe and column name
    return sd.Sort_Dataframeby_Month(df,colname)    #returning sorted order of month


# In[172]:


final=sort_data(final,'Month') #Calling the function and giving parameters
final                  #printing the table


# In[173]:


final.columns                   #Checking column names


# In[174]:


px.line(final,x='Month',y=['Price_for_resort_hotel', 'Price_for_city_hotel'],title='Room price per night over the months')           #plotting table using line


# # Analysing Demand of Hotels

# In[175]:


data_resort.head()         #watching dataframe


# In[176]:


rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()     #accessing feature and converting into data frame
rush_resort.columns=['Month','No of guests']              #renaming column name
rush_resort                                       #printing the data frame


# In[177]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()     #accessing feature and converting into data frame
rush_city.columns=['Month','No of guests']              #renaming column name
rush_city                                       #printing the data frame


# In[178]:


final_rush=rush_resort.merge(rush_city,on='Month')     #Merging rush resort with rush city


# In[179]:


final_rush.columns=['Month','No of guests in resort','No of guests of city hotel']   #Renaming new column name


# In[180]:


final_rush=sort_data(final_rush,'Month')       #Sorting month column in hierarchical manner
final_rush


# In[181]:


final_rush.columns   #Retrieving column name


# In[182]:


px.line(final_rush,x='Month',y=['No of guests in resort', 'No of guests of city hotel'],title='Total number of guests per month')


# In[183]:


data.head()


# In[184]:


data.corr()             #finding correlation


# In[185]:


co_relation=data.corr()['is_canceled']       #findint correlation with respect to is_canceled
co_relation


# In[186]:


co_relation.abs().sort_values(ascending=False)        #used abs to avoid negative values and sort_values to sort the value of correlation


# In[187]:


data.groupby('is_canceled')['reservation_status'].value_counts()   #Checking reservation_status on the basis of is_canceled


# In[188]:


list_not=['days_in_waiting_list','arrival_date_year']     #excluding this feature


# In[189]:


num_features=[col for col in data.columns if data[col].dtype!='O' and col not in list_not]     #fetching numerical colums 
num_features


# In[190]:


data.columns             #showing all columns


# In[191]:


cat_not=['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']                #excluding categorical columns


# In[192]:


cat_features=[col for col in data.columns if data[col].dtype=='O' and col not in cat_not]        #list comprehension
cat_features                                                                              


# In[193]:


data_cat=data[cat_features]        #pushing cat_features to datafrmae


# In[194]:


data_cat      #executing 


# In[195]:


data_cat.dtypes


# In[196]:


import warnings
from warnings import filterwarnings      #importing filterwarnings from warnings package
filterwarnings('ignore')                 #ingnoring warnings


# In[197]:


data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])   #converting into datetime format and updating data_cat feature


# In[198]:


data_cat.drop('reservation_status_date',axis=1,inplace=True)            #dropping reservation_status_date and updating by inplace true


# In[199]:


data_cat['cancellation']=data['is_canceled']               #inserting column


# In[200]:


data_cat.head()


# In[201]:


data_cat['market_segment'].unique()      #showing unique directory of market_segment


# In[202]:


cols = data_cat.columns                             #showing colums from 0 to 8 as we don't need cancellation column
cols


# In[203]:


data_cat.groupby(['hotel'])['cancellation'].mean()       #accessing hotels


# In[204]:


for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean())        #performing mean coding for each and every feature
    print('\n')                                                  #added new line to make it more user friendly


# In[205]:


for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()        #converted into dictionary
    data_cat[col]=data_cat[col].map(dict)                                #mapping the dictionary and updating data column


# In[206]:


data_cat.head()          #for showing few rows


# In[207]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)     #concatenating in vertical fashion that's why axis=1


# In[208]:


dataframe.head()      #dataframe showing


# In[209]:


dataframe.drop('cancellation',axis=1,inplace=True)             #dropping cancellation and updating dataframe by inplace true


# In[210]:


dataframe.shape                                                #For showing shape of dataframe


# In[211]:


sas.distplot(dataframe['lead_time'])       #Making distribution plot of lead_time by accessing dataframe


# In[212]:


import numpy as np      
def handle_outlier(col):            #taking log of lead_time time for greater extent of skewness 
    dataframe[col]=np.log1p(dataframe[col])


# In[213]:


handle_outlier('lead_time')     #calling the function


# In[214]:


sas.distplot(dataframe['lead_time'])              #showing distribution plot log applied lead_time


# In[215]:


sas.distplot(dataframe['adr'])                      #distribution plot of adr


# In[216]:


handle_outlier('adr')                                         #handling outlier for adr


# In[217]:


sas.distplot(dataframe['adr'].dropna())                       #distribution plot of adr and handling missing values by dropna  


# In[218]:


dataframe.isnull().sum()            #checking null values and doing their sum


# In[219]:


dataframe.dropna(inplace=True)          #dropping null values and updating dataframe


# In[220]:


y=dataframe['is_canceled']              #predicting independent features -> is_canceled
x=dataframe.drop('is_canceled',axis=1)     #dropping is_canceled feature


# In[221]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel        #for selecting important features


# In[222]:


feature_sel_model=SelectFromModel(Lasso(alpha=.005,random_state=0))                   #Specifying lasso regression model & setting low alpha value and putting random_state=0


# In[223]:


feature_sel_model.fit(x,y)                        #fitting data to object


# In[224]:


feature_sel_model.get_support()         #getting all the values from list


# In[225]:


cols=x.columns          #all the columns


# In[226]:


selected_feat=cols[feature_sel_model.get_support()]                     #adding filters to column


# In[227]:


print('total_features {}'.format(x.shape[1]))                   #printing total features
print('selected_features {}'.format(len(selected_feat)))        #printing the selected features


# In[228]:


selected_feat                                            #printing the entire features


# In[229]:


x=x[selected_feat]                     #updating independent dataframe


# In[230]:


x


# In[231]:


from sklearn.model_selection import train_test_split         #for splitting data into train and test set


# In[232]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)           #taking 25% of data for testing


# In[233]:


from sklearn.linear_model import LogisticRegression             #importing logisticregression


# In[234]:


logreg=LogisticRegression()                                     #calling the logisticregression class


# In[235]:


logreg.fit(X_train,y_train)                                     #fitting training data


# In[236]:


y_pred=logreg.predict(X_test)                                   #doing prediction on test data


# In[237]:


y_pred                                                          #printing the prediction array


# In[238]:


from sklearn.metrics import confusion_matrix                    #importing confusion matrix


# In[239]:


confusion_matrix(y_test,y_pred)                                 #confusion matrix of this logistic regression model


# In[240]:


from sklearn.metrics import accuracy_score                  #importing accuracy_score to check accuracy


# In[241]:


accuracy_score(y_test,y_pred)                               #checking accuracy_score of y test and prediction


# In[242]:


from sklearn.model_selection import cross_val_score           #importing cross validation


# In[243]:


score=cross_val_score(logreg,x,y,cv=10)                          #applying cross validation for achieving more accurate score


# In[244]:


score.mean()                                                    #achieved new score by calling mean


# In[245]:


from sklearn.naive_bayes import GaussianNB                  #importing gussain from navie_bayes algo


# In[246]:


from sklearn.linear_model import LogisticRegression         #importing logisticregression from linear_model 


# In[247]:


from sklearn.neighbors import KNeighborsClassifier          #importing knn algorithm


# In[248]:


from sklearn.ensemble import RandomForestClassifier         #importing random forest from ensemble


# In[249]:


from sklearn.tree import DecisionTreeClassifier             #importing decision tree classifier


# In[262]:


models=[]                                                      #this blank list is for appending all algorithms

models.append(('LogisticRegression',LogisticRegression()))     #appending logisticregression and initializing it
models.append(('Navie Bayes',GaussianNB()))                      #appending Naive Bayes and initializing it
models.append(('RandomForest',RandomForestClassifier()))         #appending random forest and initializing it
models.append(('Decision Tree',DecisionTreeClassifier()))      #appending decision tree and initializing it
models.append(('KNN',KNeighborsClassifier()))                  #appending knn and initializing it


# In[264]:


for name,model in models:                                      #iterating over models
    print(name)                                                #printing name of model
    model.fit(X_train,y_train)                             #fitting train set of x & y into model
    predictions=model.predict(X_test)                    #predicting test set of x
    
    from sklearn.metrics import confusion_matrix                    #importing confusion matrix
    print(confusion_matrix(predictions,y_test))                  #printing confusion matrix
    print('\n')
    print(accuracy_score(predictions,y_test))                   #printing accuracy_score
    print('\n')


# In[ ]:





# In[ ]:




