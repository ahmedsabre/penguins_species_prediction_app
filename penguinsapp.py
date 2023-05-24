import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
menu=st.sidebar.radio('menu',['Home','Visualization','Prediction'])
penguins=pd.read_csv('penguins.csv')
if menu=='Home':
    st.title(" penguins species Prediction App")

if menu=='Visualization':
    st.header(" Data Visualization")
    graph=st.selectbox('Different types of graphs',['scatter plots','bar graphs','histograms'])
    if graph=='histograms':
        st.subheader('bill length distribution')
        plt.figure(figsize=(16,8))
        fig=px.histogram(penguins, x="bill_length_mm")
        st.plotly_chart(fig)

        st.subheader('bill depth distribution')
        plt.figure(figsize=(16,8))
        fig=px.histogram(penguins, x="bill_depth_mm")
        st.plotly_chart(fig)

        st.subheader('flipper length distribution')
        plt.figure(figsize=(16,8))
        fig=px.histogram(penguins, x="flipper_length_mm")
        st.plotly_chart(fig)

        st.subheader('body mass distribution')
        plt.figure(figsize=(16,8))
        fig=px.histogram(penguins, x="body_mass_g")
        st.plotly_chart(fig)

    if graph=='bar graphs':
        st.subheader('species types count')
        fig=plt.figure(figsize=(16,8))
        sns.countplot(data=penguins,x='species')
        plt.xlabel('species',fontsize=20)
        plt.ylabel('count',fontsize=20)
        st.pyplot(fig)

        st.subheader('sex types count')
        fig=plt.figure(figsize=(16,8))
        sns.countplot(data=penguins,x='sex')
        plt.xlabel('sex',fontsize=20)
        plt.ylabel('count',fontsize=20)
        st.pyplot(fig)

        st.subheader('island types count')
        fig=plt.figure(figsize=(16,8))
        sns.countplot(data=penguins,x='island')
        plt.xlabel('island',fontsize=20)
        plt.ylabel('count',fontsize=20)
        st.pyplot(fig)

        plt.figure(figsize=(16,8))
        fig = px.bar(penguins, x="sex", y="body_mass_g", color='species',barmode='group')
        st.plotly_chart(fig)

    if graph=='scatter plots':
        select=st.selectbox('select an option',['species','island','sex'],key='A')
        st.subheader('relation between bill length and bill depth')
        plt.figure(figsize=(15,8))
        fig=px.scatter(data_frame=penguins,x='bill_length_mm',y='bill_depth_mm',color=select)
        plt.xlabel('bill length',fontsize=15)
        plt.ylabel('bill depth',fontsize=15)
        st.plotly_chart(fig) 

        select=st.selectbox('select an option',['species','island','sex'],key='B')
        st.subheader('relation between flipper length and body mass')
        plt.figure(figsize=(15,8))
        fig=px.scatter(data_frame=penguins,x='flipper_length_mm',y='body_mass_g',color=select)
        plt.xlabel('flipper length',fontsize=15)
        plt.ylabel('body mass',fontsize=15)
        st.plotly_chart(fig)   

if menu=='Prediction': 
    df=penguins.drop('sex',axis=1)
    df=df.copy() 
    target='species' 
    encode=['island']
    for col in encode:
        dummy=pd.get_dummies(df[col],prefix=col)
        df=pd.concat([df,dummy],axis=1)
        del df[col]

    target_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}  
    def target_encode(val):
        return target_mapper[val]
    df['species']=df['species'].apply(target_encode)
    x=df.drop('species',axis=1)
    y=df['species']
    clf=RandomForestClassifier()
    clf.fit(x,y)
    
    st.sidebar.header('user inputs')
    def user_inputs():
        island=st.sidebar.selectbox('Island', ('Biscoe','Dream','Torgersen'))
        bill_length_mm=st.sidebar.slider('Bill_Length(mm)',15.0,70.0)
        bill_depth_mm=st.sidebar.slider('Bill_Depth(mm)',10.0,30.0)
        flipper_length_mm=st.sidebar.slider('Flipper_Length(mm)',120.0,250.0)
        body_mass_g=st.sidebar.slider('Body_Mass(g)',1000.0,10000.0)
        data={'island':island,'bill_length_mm':bill_length_mm,'bill_depth_mm':bill_depth_mm,'flipper_length_mm':flipper_length_mm,'body_mass_g':body_mass_g}
        features=pd.DataFrame(data,index=[0])
        return features
    df=user_inputs()

    penguins_raw=pd.read_csv('penguins.csv')
    penguins_raw=penguins_raw.drop('sex',axis=1)
    penguins=penguins_raw.drop(columns=['species'])
    df=pd.concat([df,penguins],axis=0)
    encode=['island']
    for col in encode:
        dummy=pd.get_dummies(df[col],prefix=col)
        df=pd.concat([df,dummy],axis=1)
        del df[col]
    df=df[:1] 
    st.subheader('user_inputs')
    st.write(df)
    prediction=clf.predict(df)
    prediction_proba=clf.predict_proba(df)
    st.subheader('species names')
    st.write(target_mapper)
    st.subheader('prediction')
    penguins_species=np.array(['Adelie','Chinstrap','Gentoo'])
    st.write(penguins_species[prediction])
    st.subheader('prediction probability')
    st.write(prediction_proba)







    
