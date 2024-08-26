# Rekommend

Rekommend is a personalized music recommendation system designed to help users discover new songs based on their preferences. The system uses an autoencoder neural network to encode and analyze song features, providing accurate song recommendations. This README will guide you through the project’s setup, usage, and architecture.

## Features

- **Personalized Recommendations:** Provides song suggestions based on a selected song’s features.
- **Search Functionality:** Allows users to search for songs by name and choose from a list of results.
- **Colorful Command-Line Interface:** Engages users with a visually appealing and intuitive text interface.


## Setup

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/rekommend.git
   cd rekommend
### 2. Install Dependencies

Create a virtual environment and install the required packages:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt

### 3. Download Song Data

Ensure you have the `song_data.csv` file in the project directory. This file should contain song information including:
Check out Moosic-Rekom [dataset](https://github.com/Pradevel/MusicRekom)
- Name
- Artist
- Danceability
- Energy
- Keywords

## Usage

### 1. Run the Application
Execute the main script to start the application:

    ```bash
    python main.py

### 2. Interact with the System

- **Search for Songs:** Enter a song name when prompted. The system will display a list of matching songs.
- **Select a Song:** Choose a song by entering the corresponding number. The system will provide recommendations based on your selection.
- **Exit:** Type ‘exit’ to quit the application.

## Architecture

The Rekommend system consists of the following components:

1. **Data Processing:**
   - `data_processing.py`: Handles loading and preprocessing of song data. Converts raw data into a format suitable for model training and recommendation.
   
2. **Model:**
   - `model.py`: Defines the autoencoder neural network. Includes functions to build, train, encode features, and save/load the model.
   
3. **Recommendation:**
   - `recommendation.py`: Contains functions to recommend songs based on a user’s choice. Uses encoded features to find similar songs.
   
4. **Utilities:**
   - `utils.py`: Provides helper functions for interacting with the song dataset, such as searching songs by name.
   
5. **Main Application:**
   - `main.py`: The main script that integrates all components. Provides a command-line interface (CUI) for users to interact with the system.


