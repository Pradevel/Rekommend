import os
import sys
from colorama import Fore, Style, init

# Initialize Colorama
init()

def print_intro_logo():
    logo = f"""
{Fore.CYAN + Style.BRIGHT}
  _______    _______  __   ___   ______   ___      ___  ___      ___   _______  _____  ___   ________   
 /"      \  /"     "||/"| /  ") /    " \ |"  \    /"  ||"  \    /"  | /"     "|(\"   \|"  \ |"      "\  
|:        |(: ______)(: |/   / // ____  \ \   \  //   | \   \  //   |(: ______)|.\\   \    |(.  ___  :) 
|_____/   ) \/    |  |    __/ /  /    ) :)/\\  \/.    | /\\  \/.    | \/    |  |: \.   \\  ||: \   ) || 
 //      /  // ___)_ (// _  \(: (____/ //|: \.        ||: \.        | // ___)_ |.  \    \. |(| (___\ || 
|:  __   \ (:      "||: | \  \\        / |.  \    /:  ||.  \    /:  |(:      "||    \    \ ||:       :) 
|__|  \___) \_______)(__|  \__)\"_____/  |___|\__/|___||___|\__/|___| \_______) \___|\____\)(________/  
{Style.RESET_ALL}
{Fore.YELLOW}Welcome to Rekommend - Your Personal Music Recommendation System!{Style.RESET_ALL}
    """
    print(logo)

print_intro_logo()

from keras import Model
from data_processing import load_and_preprocess_data
from model import build_autoencoder, train_autoencoder, encode_features, save_autoencoder, load_autoencoder
from recommendation import recommend_songs_by_name
from utils import get_songs_by_name

MODEL_PATH = "model.keras"
df, combined_features = load_and_preprocess_data('song_data.csv')

if os.path.exists(MODEL_PATH):
    print(Fore.GREEN + "Loading the existing model..." + Style.RESET_ALL)
    autoencoder = load_autoencoder(MODEL_PATH)
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.layers[7].output)
else:
    print(Fore.YELLOW + "Model not found. Building and training a new model..." + Style.RESET_ALL)
    autoencoder, encoder = build_autoencoder(input_dim=combined_features.shape[1])
    train_autoencoder(autoencoder, combined_features)
    save_autoencoder(autoencoder, path=MODEL_PATH)
    print(Fore.GREEN + "Model trained and saved." + Style.RESET_ALL)

encoded_features = encode_features(encoder, combined_features)

def print_recommendations(recommendations):
    print(Fore.CYAN + "\nRecommended Songs:" + Style.RESET_ALL)
    print(Fore.CYAN + "-" * 40 + Style.RESET_ALL)
    for i, rec in recommendations.iterrows():
        print(Fore.LIGHTGREEN_EX + f"{i + 1}. Song Name: {rec['name']}" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + f"   Artist(s): {rec['artists']}" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 40 + Style.RESET_ALL)


def search_and_choose_song():
    while True:
        search_query = input(Fore.YELLOW + "\nEnter the song name (or type 'exit' to quit): " + Style.RESET_ALL).strip()

        if search_query.lower() == 'exit':
            print(Fore.RED + "Goodbye!" + Style.RESET_ALL)
            sys.exit()

        songs = get_songs_by_name(df, search_query)

        if not songs.empty:
            print(Fore.GREEN + "\nFound Songs:" + Style.RESET_ALL)
            for i, (index, song) in enumerate(songs.iterrows(), 1):
                print(Fore.LIGHTMAGENTA_EX + f"{i}. {song['name']} by {song['artists']}" + Style.RESET_ALL)

            try:
                choice = int(input(Fore.YELLOW + "\nSelect a song by entering the corresponding number: " + Style.RESET_ALL))

                if 1 <= choice <= len(songs):
                    selected_song = songs.iloc[choice - 1]
                    print(Fore.GREEN + f"\nYou selected: {selected_song['name']} by {selected_song['artists']}" + Style.RESET_ALL)
                    return selected_song
                else:
                    print(Fore.RED + "Invalid selection. Please choose a valid number." + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "Please enter a valid number." + Style.RESET_ALL)
        else:
            print(Fore.RED + f"Sorry, no songs found matching '{search_query}'." + Style.RESET_ALL)


def recommend_and_display(selected_song):
    recommendations = recommend_songs_by_name(df, encoded_features, selected_song['name'])
    if recommendations is not None and not recommendations.empty:
        print_recommendations(recommendations)
    else:
        print(Fore.RED + f"No recommendations found for {selected_song['name']}." + Style.RESET_ALL)


while True:
    selected_song = search_and_choose_song()
    if selected_song is not None:
        recommend_and_display(selected_song)