import streamlit as st
import warnings
import numpy as np
import pandas as pd
import os
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import difflib
from sklearn import preprocessing
from scipy.spatial import distance
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# Load data
file_path = 'spotify_data.csv'
df = pd.read_csv(file_path)
df = df.drop('Unnamed: 0', axis=1)

# Prepare data
df = df.drop(['time_signature', 'key'], axis=1)
df.drop_duplicates(subset=['track_id'], inplace=True)
scaler = preprocessing.MinMaxScaler()
numerical_cols = df.select_dtypes(include=np.number).columns
data_norm = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols, index=df['track_id'])

# Music Recommendation System After Listening To A Song (Euclidean-based)
def recommend_by_euclidean(track_name):
    track_id = df[df['track_name'] == track_name][['track_id']]
    
    if track_id.empty:
        return "Track not found!"
    
    track_id = track_id.values[0][0]
    target_track = list(data_norm.loc[track_id])

    data_result = pd.DataFrame()
    data_result['euclidean'] = [distance.euclidean(obj, target_track) for index, obj in data_norm.iterrows()]
    data_result['track_id'] = data_norm.index

    data_rec = data_result.sort_values(by=['euclidean']).iloc[:6]

    data_init = df.set_index(df.loc[:, 'track_id'])
    track_list = pd.DataFrame()
    for i in list(data_rec.loc[:, 'track_id']):
        if i in list(df.loc[:, 'track_id']):
            track_info = data_init.loc[[i], ['track_name', 'artist_name']]
            track_list = pd.concat([track_list, track_info], ignore_index=True)
        
    return track_list

# Music Recommendation System Using Listening History (Clustering-based)
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=30, verbose=False))], verbose=False)

X = df.select_dtypes(np.number)
song_cluster_pipeline = song_cluster_pipeline.fit(X)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['track_name'] == song['name']) & 
                                 (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return None

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[numerical_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def recommend_songs(song_list, spotify_data, n_songs=10):
    numerical_cols = song_cluster_pipeline.steps[0][1].feature_names_in_
    spotify_data_aligned = spotify_data[numerical_cols]
    
    # Calculate the center vector from the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data_aligned)
    song_center_scaled = scaler.transform(song_center.reshape(1, -1))

    # Calculate the distance and choose the closest songs
    distances = np.linalg.norm(scaled_data - song_center_scaled, axis=1)
    recommendations = spotify_data.iloc[np.argsort(distances)[:n_songs]]

    # Format the results
    formatted_recommendations = []
    for _, row in recommendations.iterrows():
        song_entry = f"{row['track_name']} - {row['year']} - {row['artist_name']}"
        formatted_recommendations.append(song_entry)
    
    return formatted_recommendations

# Check song in database (ex, DataFrame df)
def check_song_in_database(song_name, song_year):
    song_in_db = df[(df['track_name'] == song_name) & (df['year'] == song_year)]
    return not song_in_db.empty  # if song exists, return True


# Streamlit UI
st.title("Spotify Song Recommendation System")

# Sidebar for Recommendation Type Selection
st.sidebar.header("Recommendation Type")
option = st.sidebar.radio("Choose a recommendation type:", ["1. Music Recommendation System After Listening To A Song", "2. Music Recommendation System Using Listening History"])

## 1. Recommendation type based on Euclidean
if option == "1. Music Recommendation System After Listening To A Song":
    # Enter a keyword to search for a song
    search_query = st.text_input("Enter a keyword to search for a song:")
    
    # Filter the list of songs based on the keyword
    if search_query:
        filtered_songs = df[df['track_name'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_songs = df  # If nothing entered, show all songs
    
    # Dropdown to display results after filtering
    song_name_euclidean = st.selectbox(
        "Select a song:", 
        options=filtered_songs['track_name'].unique()
    )
    
    if st.button("Get Recommendations"):
        if song_name_euclidean:
            st.subheader(f"Recommended Songs for '{song_name_euclidean}'")
            recommended_songs_euclidean = recommend_by_euclidean(song_name_euclidean)
            
            if isinstance(recommended_songs_euclidean, pd.DataFrame):
                for _, row in recommended_songs_euclidean.iterrows():
                    st.write(f"- {row['track_name']} by {row['artist_name']}")
            else:
                st.error(recommended_songs_euclidean)  # Show error if song is not found
        else:
            st.error("Please select a valid song.")


# 2. For Cluster-based recommendation (Multiple songs)
elif option == "2. Music Recommendation System Using Listening History":
    num_songs_input = st.text_input("Enter the number of songs you want to add (1-10):", value="1")

    if num_songs_input.strip() == "":
        num_songs = None
        st.error("Please enter a valid number of songs between 1 and 10.")
    else:
        try:
            num_songs = int(num_songs_input)
            if num_songs < 1 or num_songs > 10:
                raise ValueError("The number of songs must be between 1 and 10.")
        except ValueError as e:
            st.error(f"Error: {e}")
            num_songs = None

    if num_songs is not None and 1 <= num_songs <= 10:
        song_list = []  # List to store the entered songs
        is_valid = True  # Variable to check the validity of the entered data

        for i in range(num_songs):
            st.subheader(f"Song {i+1}")

            # Enter a keyword to search for a song
            search_query = st.text_input(f"Enter a keyword for song {i+1}:", key=f"search_{i+1}")
            
            # Filter the list of songs based on the keyword
            if search_query:
                filtered_songs = df[df['track_name'].str.contains(search_query, case=False, na=False)]
            else:
                filtered_songs = df
            
            # Dropdown to select song after filtering
            song_name = st.selectbox(
                f"Select the name of song {i+1}:", 
                options=filtered_songs['track_name'].unique(), 
                key=f"song_{i+1}"
            )
            
            # Dropdown to select the release year
            filtered_years = filtered_songs['year'].unique() if not filtered_songs.empty else []
            song_year = st.selectbox(
                f"Select the release year of song {i+1}:", 
                options=filtered_years if len(filtered_years) > 0 else range(1900, 2025),
                key=f"year_{i+1}"
            )

            # Validate and add the song to the list
            if song_name and song_year:
                song_exists = check_song_in_database(song_name, song_year)
                
                if song_exists:
                    song_list.append({'name': song_name, 'year': song_year})
                else:
                    st.error(f"Warning: '{song_name}' from {song_year} does not exist in the database.")
                    is_valid = False
                    break

        if st.button("Get Recommendations") and is_valid:
            if song_list:
                st.subheader("Song recommendations based on the songs you provided:")

                recommended_songs_cluster = recommend_songs(song_list, spotify_data=df)
                for song in recommended_songs_cluster:
                    st.write(f"- {song}")
            else:
                st.error("Please select at least one valid song.")
