import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Recommendations :video_game:')

"""
## How it works:

Choose a game from the dropdown below, then choose a genre and see your recommendations.

The data in use was partially taken from [here] (https://www.kaggle.com/nikdavis/steam-store-games), while the game descriptions were webscraped independently.

----
"""

# read in csv
@st.cache
def load_data():

    descriptions = pd.read_csv('app_descriptions.csv')
    data = pd.read_csv('steam.csv')
    data.rename(columns={'name':'title'}, inplace=True)
    df = data.merge(descriptions, on='title')
    df.dropna(inplace=True)

    return df

df = load_data()

# Helper function to look up game TFIDF score by title
def get_game_by_title(title, tfidf_scores, keys):
    row_id = keys[title]
    row = tfidf_scores[row_id,:]
    return row

# instantiate vectorizer
to_vec = TfidfVectorizer()

index = 0
keys = {}

df_description = df[['title', 'tags']]

for game in df_description.itertuples():
    key = game[1]
    keys[key] = index
    index += 1

# fitting vectorizer
to_vec.fit(df_description['tags'].fillna(''))

# transforming data
tfidf_scores = to_vec.transform(df_description['tags'].fillna(''))

def recommend_content(title, tfidf_scores, games):
    
    # dataframe for result storage
    recommended = pd.DataFrame(columns=['title','similarity'])
    
    # use helper function to find game
    game1 = get_game_by_title(title, tfidf_scores, keys)
    
    # loop through all games and get similarities
    for i in games['title']:
        
        # find similarity
        game2 = get_game_by_title(i, tfidf_scores, keys)
        sim_score = cosine_similarity(game1,game2)
        recommended.loc[len(recommended)] = [i, sim_score[0][0]]
    
    # returning dataframe with similarity scores and titles
    result = recommended.sort_values(by=['similarity'], ascending=False)[1:].reset_index(drop=True)
    
    return result

all_tags = []

for i in df['tags'].str.split(', '):
    try:
        for j in i:
            if j not in all_tags:
                all_tags.append(j)
    except:
        pass

all_tags = sorted(all_tags)

# select a game to base recommendation on
selected_game = st.selectbox('Select a game', df['title'])

# output game must contain this tag
selected_genre = st.selectbox('select a tag', all_tags)

# button to start recommendation
if st.button('Recommend'):
    # creating dataframe to input into function containing only games with selected tags
    games_to_compare = df.loc[(df['tags'].str.find(selected_genre) != -1) & (df['positive_ratings'] >= 1500)]

    # calculating recommendations
    recommended_games = recommend_content(selected_game, tfidf_scores, games_to_compare)

    # adding positive and negative review counts along with price 
    st.dataframe(recommended_games.head(10))






