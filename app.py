import streamlit as st
import cv2
from predict import predict_expression

st.title("Handwritten Math Solver")

uploaded = st.file_uploader("Upload an image of handwritten math expression", type=["png","jpg","jpeg"])

if uploaded:
    with open("input.png", "wb") as f:
        f.write(uploaded.read())

    st.image("input.png", caption="Uploaded Image")

    expr = predict_expression("input.png")

    st.write("### Recognized Expression:")
    st.code(expr)

    try:
        result = eval(expr)
        st.write("### Result:")
        st.success(result)
    except:
        st.error("Could not solve expression")
