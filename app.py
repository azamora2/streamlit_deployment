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
    url1 = 'https://bucketdemandaespanya.s3.amazonaws.com/index.html'
    url2 = 'https://bucketdemandaespanya.s3.amazonaws.com/project1.html'
    url3 = 'https://bucketdemandaespanya.s3.amazonaws.com/project2.html'
    url4 = 'http://54.163.48.94:8501/'

    streamlit.markdown(f'''
    <style>
        .center {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 20%;
    }}
    .center2 {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 10%;
    }}
    .center3 {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 70%;
    }}

    .navbar {{
        color: limegreen;
    }}
    
    .navbar li {{
        display: inline-block;
        height: 100%;
        line-height: 500%;
        font-size: 80%;
        border: 80% dotted #555;
        border-bottom: 80% solid lightblue;
        min-width: 10%;
    }}
    
    .navbar a {{
        text-decoration: none;
        color: white;
        display: block;
    }}
    
    .navbar a:hover {{
        background-color: lightblue;
        color: white;
    }}
    
    .navbar ul {{
        background-color: navy;
        list-style: none;
        text-align: center;
        padding: 0;
        margin: 0;
    }}
    title,body{{
    font-family: 'Courier New', Courier, monospace;
    background-color: rgb(165, 164, 164);
    }}
    html{{
    background-color: navy;
    }}

    body {{
    border: 0px;
    margin: 0px;
    padding: 0px;
    background-color: white;
    }}
    iframe{{
    display: block;
    margin-left: auto;
    margin-right: auto;
    padding:0;
    border:none;
    overflow:hidden;
    }}
    h1{{
    text-align: center;
    background-color: navy;
    color: white;
    margin: 0px;
    padding: 1%;
    }}
    </style>
    <h1>Project 3: Enter a review for the movie Shrek</h1>
    <div class="navbar">
      <ul>
        <li><a href={url1}>Home</a></li>
        <li><a href={url2}>Project 1</a></li>
        <li><a href={url3}>Project 2</a></li>
        <li><a href="http://54.163.48.94:8501/">Project 3</a></li>
      </ul>
    </div>
    ''',
    unsafe_allow_html=True)
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


    
