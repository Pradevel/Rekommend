import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    features = df[['artists', 'danceability', 'energy', 'keywords']].copy()
    features['keywords'] = features['keywords'].fillna('')
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000, output_sequence_length=100)
    vectorizer.adapt(features['keywords'])
    keywords_matrix = vectorizer(tf.constant(features['keywords'].tolist()))

    scaler = StandardScaler()
    numerical_features = features[['danceability', 'energy']]
    scaled_numerical_features = scaler.fit_transform(numerical_features)
    combined_features = np.hstack([keywords_matrix.numpy(), scaled_numerical_features])
    return df, combined_features