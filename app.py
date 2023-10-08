import streamlit as st
import joblib

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
