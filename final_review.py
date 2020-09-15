# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:08:15 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


extracted_data = pd.read_csv("D:\\AmazonProjFiles\\lifebuoyReviews.csv",encoding = "ISO-8859-1")
extracted_data.head
extracted_data.isnull().sum()#754
extracted_data.dropna(inplace=True)
extracted_data.isnull().sum()
extracted_data.head#752
extracted_data=extracted_data.drop(['CutomerName'], axis=1)
extracted_data.head
ext_rev_string=extracted_data['Reviews']
ext_rev_string=re.sub("[^A-Za-z" "]+"," ",str(ext_rev_string)).lower()
ext_rev_string=re.sub("[0-9" "]+"," ",str(ext_rev_string)) 
ip_reviews_words = ext_rev_string.split(" ")
with open("D:\\ExcelR Data\\Assignments\\Text Mining\\stop.txt","r") as sw:
        stopwords = sw.read()
        stopwords=stopwords.split("\n")        
        ip_reviews_words=[w for w in ip_reviews_words if w not in stopwords]
        ext_rev_string = " ".join(ip_reviews_words)
        
#calculating sentiment polarity and Subjectivity
from textblob import TextBlob        
polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity  
extracted_data['polarity'] = extracted_data['Reviews'].apply(polarity)
extracted_data['subjectivity'] = extracted_data['Reviews'].apply(subjectivity)
extracted_data  
# creating a function to compute the negative, neutral and positive analysis

def getAnalysis(score):
    if score < 0:
        return '-1'
    elif score ==0:
        return '0'
    else:
        return '+1'
extracted_data['sentiment'] = extracted_data['polarity'].apply(getAnalysis)
print(extracted_data.sentiment.value_counts())
print(extracted_data.sentiment.value_counts(normalize=True) * 100)
extracted_data.head

#bar plot
fig, ax = plt.subplots(figsize=(5, 5))
extracted_data['sentiment'].value_counts(normalize=True).plot(kind='bar'); 
ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
ax.set_ylabel("Percentage")
plt.show()

extracted_data.to_csv(r'D:\\AmazonProjFiles\\lifebuoyReviews_new.csv',index=False,header=True)  
#####

####NSE data
nse_data = pd.read_csv("D:\\AmazonProjFiles\\01-01-2020-TO-10-08-2020HINDUNILVRALLN.csv")
nse_data.head
nse_data.isnull().sum()
nse_3months= nse_data.iloc[82:,]
nse_3months.columns
from sklearn.preprocessing import MinMaxScaler
nse_3months['ClosePrice'] = MinMaxScaler().fit_transform(nse_3months['Close Price'].values.reshape(-1,1))
minMax = MinMaxScaler()
nse_3months['Date'] = pd.to_datetime(nse_3months['Date'])

extracted_data.shape
extracted_data.info()
df_review = pd.DataFrame(columns= ['polarity','subjectivity','sentiment'])
df_review[['polarity','subjectivity','sentiment']] = extracted_data[['polarity','subjectivity','sentiment']].astype(float)
df_review.index = pd.to_datetime(extracted_data['Date'])
df_review = df_review.resample('D').mean().ffill()

df_review['Date'] = df_review.index
df_review.reset_index(inplace= True,drop=True)
merge_all_reviews = pd.merge(nse_3months,df_review,on='Date')

## single plot
x= merge_all_reviews['Date']
y= merge_all_reviews['polarity']
plt.figure(figsize=(30,10))
plt.plot(x,y ,color='blue')
plt.title('date vs polarity',fontsize=28)
plt.xlabel('stock Date',fontsize=28)
plt.ylabel('polarity',fontsize=28)
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.show()



#####
import matplotlib.pyplot as plt
x = merge_all_reviews['Date']
plt.plot(x,merge_all_reviews['ClosePrice'], label='ClosePrice')
plt.plot(x,merge_all_reviews['polarity'], label='Polarity')
plt.legend(loc='best')
plt.show()
