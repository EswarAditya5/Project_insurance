#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#get_ipython().run_line_magic('cd', '"D:\\Imarticus\\Machine Learning\\lms\\projects\\Linear Regression Project\\Dataset"')


# In[3]:


insurance=pd.read_csv('insurance.csv')


# In[4]:


insurance.head()


# In[5]:


insurance.info()


# In[6]:


insurance.describe()


# In[7]:


insurance.age.value_counts()


# In[8]:


insurance.region.value_counts()


# In[9]:


numcols=insurance.select_dtypes(include=np.number)
objcols=insurance.select_dtypes(include='object')


# In[10]:


numcols.head()


# In[11]:


objcols.head()


# In[12]:


sns.heatmap(numcols.corr(),annot=True)


# In[13]:


#combinedf.replace({False:0,True:1},inplace=True)
objcols.replace({'female':0,'male':1},inplace=True)
objcols.replace({'yes':1,'no':0},inplace=True)
objcols.replace({'southeast':0,'southwest':1,'northwest':2,'northeast':3},inplace=True)


# In[14]:


combinedf=pd.concat([numcols,objcols],axis=1)


# In[15]:


combinedf.head()


# In[16]:


X=combinedf[['age','bmi','children','sex','smoker','region']]
y=combinedf.charges


# In[17]:


fig,ax=plt.subplots(3,1)
y.plot(kind='bar',ax=ax[0])
sns.boxplot(y,ax=ax[1],orient='h')
y.plot(kind='density',ax=ax[2])


# In[18]:


np.log(y).plot(kind='density')


# # Linear Regression

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


rig=LinearRegression()


# In[21]:


rigmodel=rig.fit(X,y)


# In[22]:


rigmodel.score(X,y)


# In[23]:


rigpredict=rigmodel.predict(X)


# In[24]:


resid=y-rigpredict


# In[25]:


np.sqrt(np.mean(resid**2))


# In[26]:


import streamlit as st
import joblib


# In[27]:


joblib.dump(rig,'rigmodel.sav')


# In[ ]:





# # Decision Tree

# In[28]:


from sklearn.tree import DecisionTreeRegressor


# In[29]:


tree=DecisionTreeRegressor(max_depth=8)


# In[30]:


treemodel=tree.fit(X,y)


# In[31]:


treemodel.score(X,y)


# In[32]:


treepredict=tree.predict(X)


# In[33]:


treeresid=y-treepredict


# In[34]:


np.sqrt(np.mean(treeresid**2))


# In[62]:


joblib.dump(tree,'treemodel.sav')


# In[ ]:





# # Random Forest

# In[36]:


from sklearn.ensemble import RandomForestRegressor


# In[37]:


rf=RandomForestRegressor(max_depth=8)


# In[38]:


rfmodel=rf.fit(X,y)


# In[39]:


rfmodel.score(X,y)


# In[40]:


rfpredict=rf.predict(X)


# In[41]:


rfresid=y-rfpredict


# In[42]:


np.sqrt(np.mean(rfresid**2))


# In[43]:


joblib.dump(rf,'rfmodel.sav')


# In[ ]:





# # GBM

# In[44]:


from sklearn.ensemble import GradientBoostingRegressor


# In[45]:


gbm=GradientBoostingRegressor()


# In[46]:


gbmmodel=gbm.fit(X,y)


# In[47]:


gbmmodel.score(X,y)


# In[48]:


gbmpredict=gbm.predict(X)


# In[49]:


gbmresid=y-gbmpredict


# In[50]:


np.sqrt(np.mean(gbmresid**2))


# In[51]:


joblib.dump(gbm,'gbmmodel.sav')


# In[ ]:





# # SVR

# In[52]:


from sklearn.svm import SVR


# In[53]:


svr=SVR()


# In[54]:


svrmodel=svr.fit(X,y)


# In[55]:


svrmodel.score(X,y)


# In[ ]:

st.title('Predict charges')
st.markdown('Model Predict Charges')
st.header('Person Details')


# In[57]:


col1,col2,col3,col4,col5,col6=st.columns(6)
with col1:
    age = st.slider('age',2,100,1)
with col2:
    sex = st. selectbox('sex',['Male','Female'])
with col3:
    bmi = st.number_input('bmi')
with col4:
    children = st.selectbox('children',[0,1,2,3,4,5])
with col5:
    smoker = st.selectbox('smoker',['yes','no'])
with col6:
    region = st.selectbox('region',['southeast','southwest','northwest','northeast'])


# In[58]:
choice=st.radio('select model',[rig,tree,rf,gbm])
if st.button('Predict Charges'):
    selected_model = choice
    selected_model.fit(X, y)
    result=choice.predict(np.array(X))
    st.text(result[0])

# In[63]:


def predict(data):
    clf=joblib.load(choice)
    return clf.predict(data)




# In[ ]:



# In[ ]:
#choice=st.radio('select model',[rig,tree,rf,gbm])

#if st.button('Predict Charges'):
#    result=choice.predict(np.array(X))
#   st.text(result[0])




# In[ ]:




