import streamlit as st
import requests

backend = "http://fastapi:8000/predict"

KEYS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
st.set_page_config(layout="wide")

st.title("IRIS Classification Task")

feature1 = st.slider("Select the sepal length (cm)", 4.3, 7.9, 4.5)
feature2 = st.slider("Select the sepal width (cm)", 2.0, 4.4, 2.5)
feature3 = st.slider("Select the petal length (cm)", 1.0, 6.9, 4.0)
feature4 = st.slider("Select the petal width (cm)", 0.1, 2.5, 1.5)

st.write("sepal length (cm):", feature1)
st.write("sepal width (cm):", feature2)
st.write("petal length (cm):", feature3)
st.write("petal width (cm):", feature4)

features = [feature1, feature2, feature3, feature4]

if st.button("Classify"):
    features = {KEYS[i]: features[i] for i in range(len(KEYS))}
    response = requests.post(backend, json=features)
    if response.status_code == 200:
        answer = response.json()
        st.write(f"The flower is : {answer['prediction']}")
    else:
        st.write("Error - Application is not working")
