import numpy as np
from scipy.spatial.distance import cosine

def cosine_similarity_matrix(features):
    num_features = features.shape[0]
    sim_matrix = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            sim_matrix[i, j] = 1 - cosine(features[i], features[j])
    return sim_matrix

def recommend_songs_by_track_id(df, encoded_features, song_index, top_n=5):
    similarity_matrix = cosine_similarity_matrix(encoded_features)
    similar_indices = similarity_matrix[song_index].argsort()[-top_n - 1:-1]
    recommendations = df.iloc[similar_indices]
    return recommendations


import numpy as np


def recommend_songs_by_name(df, encoded_features, song_name, top_n=5):
    song = df[df['name'].str.contains(song_name, case=False, na=False, regex=False)]
    if not song.empty:
        song_index = song.index[0]
        similarities = np.dot(encoded_features, encoded_features[song_index])
        similar_indices = similarities.argsort()[-top_n - 1:-1][::-1]
        recommendations = df.iloc[similar_indices]
        return recommendations
    return None