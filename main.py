import os
import sys

from keras import Model
from data_processing import load_and_preprocess_data
from model import build_autoencoder, train_autoencoder, encode_features, save_autoencoder, load_autoencoder
from recommendation import recommend_songs_by_name
from utils import get_songs_by_name

MODEL_PATH = "model.keras"
df, combined_features = load_and_preprocess_data('song_data.csv')

if os.path.exists(MODEL_PATH):
    print("Loading the existing model...")
    autoencoder = load_autoencoder(MODEL_PATH)
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.layers[7].output)
else:
    print("Model not found. Building and training a new model...")
    autoencoder, encoder = build_autoencoder(input_dim=combined_features.shape[1])
    train_autoencoder(autoencoder, combined_features)
    save_autoencoder(autoencoder, path=MODEL_PATH)
    print("Model trained and saved.")

encoded_features = encode_features(encoder, combined_features)

def print_recommendations(recommendations):
    print("\nRecommended Songs:")
    print("-" * 40)
    for i, rec in recommendations.iterrows():
        print(f"{i + 1}. Song Name: {rec['name']}")
        print(f"   Artist(s): {rec['artists']}")
        print("-" * 40)


def search_and_choose_song():
    while True:
        search_query = input("\nEnter the song name (or type 'exit' to quit): ").strip()

        if search_query.lower() == 'exit':
            print("Goodbye!")
            sys.exit()

        songs = get_songs_by_name(df, search_query)

        if not songs.empty:
            print("\nFound Songs:")
            for i, song in songs.iterrows():
                print(f"{i + 1}. {song['name']} by {song['artists']}")

            try:
                choice = int(input("\nSelect a song by entering the corresponding number: ")) - 1

                if 0 <= choice < len(songs):
                    selected_song = songs.iloc[choice]
                    print(f"\nYou selected: {selected_song['name']} by {selected_song['artists']}")
                    return selected_song
                else:
                    print("Invalid selection. Please choose a valid number.")
            except ValueError:
                print("Please enter a valid number.")
        else:
            print(f"Sorry, no songs found matching '{search_query}'.")


def recommend_and_display(selected_song):
    recommendations = recommend_songs_by_name(df, encoded_features, selected_song['name'])
    if recommendations is not None and not recommendations.empty:
        print_recommendations(recommendations)
    else:
        print(f"No recommendations found for {selected_song['name']}.")


while True:
    selected_song = search_and_choose_song()
    if selected_song is not None:
        recommend_and_display(selected_song)