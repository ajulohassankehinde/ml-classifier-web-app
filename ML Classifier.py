#!/usr/bin/env python
#coding: utf-8


import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

st.title("ML Classifier Web Application")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

test_size = st.sidebar.slider("Test Size", 0.1, 0.5)

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

st.write("### Summary")
st.write("The shape of the dataset:", X.shape)
st.write("The number of classes:", len(np.unique(y)))
st.write("Test Size:", test_size*100, "%")


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
#st.write("Confusion Matrix:", conf_mat)

st.write("### Heatmap")

if st.checkbox("Show Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(10,10))
    st.write(sns.heatmap(conf_mat, annot=True,linewidths=0.5, cmap = "GnBu"))
    st.pyplot(fig)

st.write("### Dataset")

if st.checkbox("Show Dataset"):
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0,value=5)
    if rows > 0:
        st.write("Explnatory Variables")
        st.dataframe(pd.DataFrame(X).head(rows))
        st.write("Response Variable")
        st.dataframe(pd.DataFrame(y).head(rows))
        
    st.write("#### Data Description")    
    st.write("Explnatory Variables")
    st.write(pd.DataFrame(X).describe())
    st.write("Response Variable")
    st.write(pd.DataFrame(y).describe())
