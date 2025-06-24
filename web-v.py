import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.title("Age & Gender Estimation")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg','jpeg','png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        result = DeepFace.analyze(np.array(img), actions=['age', 'gender'])
    
    st.success("Analysis Complete!")
    st.write(f"**Age:** {result[0]['age']}")
    st.write(f"**Gender:** {result[0]['gender']}")
