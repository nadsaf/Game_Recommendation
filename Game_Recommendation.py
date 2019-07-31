# data set : https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings

import pandas as pd
import numpy as np

dfGame = pd.read_csv('2.Video_Games_Sales_as_at_22_Dec_2016.csv')

# Missing value ---------------------------------------------------------------
# print(dfGame.isnull().sum())
dfGame = dfGame.dropna(subset= ['Genre', 'Platform']).reset_index()
dfGame = dfGame[['Name', 'Platform', 'Genre']]
# print(dfGame.head(3))

# Add new col: 'Platform + Genre' ---------------------------------------------

def mergeCol(i):
    return str(i['Platform']) + ' ' + str(i['Genre'])

dfGame['features'] = dfGame['Genre'].str.cat(dfGame['Platform'], sep = ' ')
# print(dfGame.head(10))
# print(len(dfGame['Platform'].unique())) # 31
# print(len(dfGame['Genre'].unique()))    # 12 

#============================ Count Vectorizer =================================
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer = lambda i : i.split(' ')
    )
matrixFeature = model.fit_transform(dfGame['features'])
features = model.get_feature_names()
# print(features)
# print(len(features))            # total features : 43
# print(matrixFeature.toarray())  


#================== Count Based Filter Recommendataion Sys======================
# Indeks kesamaan -------------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)
# print(score)
# print(score[0])

MainGame = 'ZombiU'
MainPlatf = 'PS4'

index = dfGame[(dfGame['Name'] == MainGame) & (dfGame['Platform'] == MainPlatf)].index.values[0]

# print(score[index][17])

# Cos similarity - All Games dari game yg disuka  ----------------------------------
daftarScore = list(enumerate(score[index]))
# print(daftarScore)
sort_DaftarScore = sorted(
    daftarScore,
    key = lambda i : i[1],  
    reverse = True          # desc
)

# cos similliarty score > 80% ---------------------------------------------------------
GameSama80up = []
for i in sort_DaftarScore:
    if i[1] > 0.8:
        GameSama80up.append(i)
# print(GameSama80up)

# Dynamic Recommendation  --------------------------------------------------------------
import random
rekomendasi = random.choices(GameSama80up, k=5)
# print(rekomendasi)
print('Game Recommendation for : ', dfGame.iloc[index]['Name'], dfGame.iloc[index]['Platform'])
for i in rekomendasi:
    print(
        dfGame.iloc[i[0]]['Name'], '|',
        dfGame.iloc[i[0]]['Platform'], '|',
        dfGame.iloc[i[0]]['Genre'])