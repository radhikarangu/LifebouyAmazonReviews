from flask import Flask, render_template
import os

gp = r"C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\app_flask\static"

app = Flask(__name__)


@app.route('/')                            

def show_index():
    return render_template("index_3.html")

@app.route('/home')
def home():
   lifebouy_sanitizer_1 = os.path.join( '/static/lifebouy_sanitizer_1.png')
   amazon_page = os.path.join( '/static/AMAZON _PAGE.png')
   extracted_data = os.path.join( '/static/extracted reviews.png')
   return render_template("index_2.html", user_image_7 = lifebouy_sanitizer_1,
                          user_image_10 = amazon_page,
                          user_image_11 = extracted_data
                          )               
@app.route('/eda')
def eda():
    wordcloud_all_reviews_1 = os.path.join( '/static/WORDCLOUD_1.png')
    wordcloud_positive_reviews_1= os.path.join( '/static/POSITIVE WORDCLOUD_1.png')
    wordcloud_negative_reviews_1= os.path.join( '/static/NEGATIVE WORDLCOUD_1.png')
    bigram_1 = os.path.join( '/static/BIGRAM.png')
    trigram_1 = os.path.join( '/static/TRIGRAM.png')
    sentiment_count_1 = os.path.join( '/static/sentiment_count_barchart_1.png')
   
    return render_template("index_1.html", user_image_1 = wordcloud_all_reviews_1,
                           user_image_2 = wordcloud_positive_reviews_1,
                           user_image_3 = wordcloud_negative_reviews_1,
                           user_image_4 = bigram_1,
                           user_image_5 = trigram_1,
                           user_image_6 = sentiment_count_1 
 )
    
@app.route('/stock')
def stock():
    
   return render_template("index_4.html")               
app.run()

