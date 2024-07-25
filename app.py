# save this as app.py
from flask import Flask, request, jsonify, render_template
import instaloader
import pandas as pd
import numpy as np

import joblib
from instaloader import Post

app = Flask(__name__)

model = joblib.load('model/sentiment_instagram.pkl')
# Data baru yang ingin diprediksi
new_data = ["tolol","bagus","jelek","semoga ibunya jahat"]

# Melakukan prediksi menggunakan model yang dimuat
predictions = model.predict(new_data)

# Menampilkan hasil prediksi
print(predictions)
L = instaloader.Instaloader()



def model_analisis_instagram(url, max_comments):
    
    shortcode = url.split('/')[-2]

    # Login ke Instagram (opsional tetapi disarankan untuk akses yang lebih luas)
    # username = 'hatiayam6'
    username = 'manusiaaaa7670'
    password = '87651234'
    L.login(username, password)
    # Ambil postingan menggunakan shortcode
    post = Post.from_shortcode(L.context, shortcode)
    
    comments_data = []
    comment_count = 0
    # Ambil komentar dari postingan hingga mencapai batas
    for comment in post.get_comments():
        comments_data.append({
            'username': comment.owner.username,
            'text': comment.text,
            'created_at_utc': comment.created_at_utc
        })
        comment_count += 1
        if comment_count >= max_comments:
            break
    
    # Konversi ke DataFrame
    df = pd.DataFrame(comments_data)
    df = df.drop(columns=[ 'created_at_utc'])
    return df
   
@app.route('/profile', methods=['POST'])
def profile():
    data = request.get_json()
    url = data.get('url')
    jumlah = data.get('jumlah')

    post = instaloader.Post.from_shortcode(L.context, url.split('/')[-2])
    user = post.owner_profile
    profile = user.profile_pic_url
    userName = post.owner_username

    data = {"name": userName,
            "profile": profile}
    return jsonify(data)

@app.route('/sentiment/ig', methods=['POST'])
def prediksi_IG():
    data = request.get_json()
    url = data.get('url')
    jumlah = data.get('jumlah')
    
    if not url or not jumlah:
        return jsonify({"error": "url and jumlah are required fields"}), 400

    try:
        # sentiment
        hasilModel = model_analisis_instagram(url, int(jumlah))
        # print(hasilModel["sentiment"])
        sentimen = model.predict(hasilModel["text"].values)
        sentimen= list(sentimen)
        positive = sentimen.count(1)
        negative = sentimen.count(-1)
        username = list(hasilModel["username"].values)
        text = list(hasilModel["text"].values)
        dataList = []
        for i in range(len(sentimen)):
            if sentimen[i] == 1 :
                dataList.append({
                "username": username[i],
                "text": text[i],
                "sentiment": "positive"
            })
            else :
                dataList.append({
                "username": username[i],
                "text": text[i],
                "sentiment": "negative"
            })
        data = {"Url": url,
                "count_coment":jumlah,
                "positive": positive,
                "negative": negative,
                "sentiment": dataList
                }
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sentiment/text', methods=['POST'])
def prediksi_sentiment():
    data = request.get_json()
    text = data.get('text')
    text = model.predict([text])
    text = list(text)
    print(type(text))
    
    if text[0] == 1 :
        sentiment = {"data": "kalimat tersebut merupakan sentimen POSITIF"}
    else :
        sentiment = {"data": "kalimat tersebut merupakan sentimen NEGATIF"}
    
    print(sentiment)
    return jsonify(sentiment)

@app.route("/")
def hello():
    return render_template("index.html")

app.run(debug=True)