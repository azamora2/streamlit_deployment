# load packages
import os
import numpy as np
import pandas as pd
import pickle
import streamlit
import re
from PIL import Image

image = Image.open('shrek2.jpg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load tokenizer for preprocessing
with open('tfid.pkl', 'rb') as tk:
    tfidf = pickle.load(tk)

with open('model.pkl', 'rb') as tk:
    model = pickle.load(tk)

def sentiment_prediction(review):
    sentiment=[]
    # Convert to array
    input_review = [review]
    input_review = [x.lower() for x in input_review]
    input_review = [re.sub('[^a-zA-z0-9\s]','',x) for x in input_review]
 

    # Convert into list with word ids
    sentiment = model.predict(tfidf.transform(input_review))[0]
    
    if(sentiment == 0):
        pred="Negative"
    else:
        pred= "Positive"
    
    return pred


def run():
    streamlit.title("Enter a review for the movie Shrek")
    html_temp="""
    
    """
 
    streamlit.markdown(html_temp)
    streamlit.image(image, caption='Shrek', width= 100,use_column_width=True)
    review=streamlit.text_input("Enter the Review ")
    prediction=""
    
    if streamlit.button("Predict Sentiment"):
        prediction=sentiment_prediction(review)
    streamlit.success("The sentiment predicted by Model : {}".format(prediction))
    
if __name__=='__main__':
    run()


    
