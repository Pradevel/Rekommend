def get_song_by_track_id(df, track_id):
    song = df[df['id'] == track_id]
    if not song.empty:
        return song.iloc[0]
    else:
        return None

def get_songs_by_name(df, song_name):
    songs = df[df['name'].str.contains(song_name, case=False, na=False)]
    return songs