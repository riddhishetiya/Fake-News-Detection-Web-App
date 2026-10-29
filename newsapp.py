import streamlit as st
import joblib as jb
st.set_page_config(
    page_title="News Detector",
    layout="centered"
)
@st.cache_resource
def load_assets():
    model = jb.load('fake_news_model.pkl')
    vectorizer=jb.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    assets_loaded = False

st.title('Fake News Detector')
st.markdown("""
This app uses a **PassiveAggressiveClassifier** and **TF-IDF Vectorization** 
to instantly analyze news headlines and articles and determine if they are **REAL** or **FAKE**.
""")
st.write("---")

user_input = st.text_area(
    "paste the news headline or article here",
    height=200,
    placeholder="type or paste your text here"
)
if st.button("Analyze News", type="primary"):
    if not user_input.strip():
        st.warning("please enter some text to analyze")
    elif assets_loaded:
        transformed_text= vectorizer.transform([user_input])
        prediction= model.predict(transformed_text)[0]

    st.write("Prediction Result: ")
    if str(prediction).upper() == 'REAL':
        st.success("The news piece provided is real")
    else :
        st.error("The news piece provided is fake")



