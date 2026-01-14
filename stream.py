import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_df = pd.read_csv('WELFake_Dataset.csv.zip')
news_df = news_df.fillna(' ')


# stemming
ps = PorterStemmer()

def stemming(title):
    stemmed_title = re.sub('[a-zA-Z]'," ",title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = " ".join(stemmed_title)
    return stemmed_title

news_df['title'] = news_df['title'].apply(stemming)


x = news_df['title'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2,stratify=y, random_state = 1)


model = LogisticRegression()
model.fit(x_train,y_train)




    #website
st.title('Fake News detector')
input_text = st.text_input('Enter news Article')
button = st. button("Check")
def prediction(input_Text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
         st.markdown("<h2 style='color:red;'>❌ FAKE NEWS</h2>", unsafe_allow_html=True)
    else:
            st.markdown("<h2 style ='color:green;'>✅ REAL NEWS</h2>", unsafe_allow_html=True)





       # st.write('❌The news is Fake')
   # else:
      #  st.markdown("<h3 style ='color:green;'>✅ The news is Real</h3>")


        # Display result with colored text
        #if prediction == 1:
           # st.markdown("<h3 style='color:red;'>❌ The news is Fake</h3>", unsafe_allow_html=True)
    # else:nb
      #      st.markdown("<h3 style ='color:green;'>✅ The news is Real</h3>", unsafe_allow_html=True)


          