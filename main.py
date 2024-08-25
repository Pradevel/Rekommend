import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
df = pd.read_csv('song_data.csv')
features = df[['artists', 'danceability', 'energy', 'keywords']].copy()
features.loc[:, 'keywords'] = features['keywords'].fillna('')

vectorizer = TfidfVectorizer()
keywords_matrix = vectorizer.fit_transform(features['keywords'])
scaler = StandardScaler()
numerical_features = features[['danceability', 'energy']]

scaled_numerical_features = scaler.fit_transform(numerical_features)
combined_features = hstack([keywords_matrix, scaled_numerical_features])
similarity_matrix = cosine_similarity(combined_features)

def recommend(song_index, top_n=5):
    similar_indices = similarity_matrix[song_index].argsort()[-top_n-1:-1]
    return df.iloc[similar_indices]

song_index = 0
recommendations = recommend(song_index)
print(recommendations[['name', 'artists']])