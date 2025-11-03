'''
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import random
import traceback # Import traceback for detailed error reporting

# --- Setup: Download NLTK VADER Lexicon ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading NLTK 'vader_lexicon' for sentiment analysis...")
    nltk.download('vader_lexicon')
    print("NLTK 'vader_lexicon' downloaded.")
except Exception as e:
    print(f"Error checking/downloading NLTK 'vader_lexicon': {e}")
    print("VADER sentiment analysis might not function correctly.")


# --- 1. Load Local Datasets ---
song_df = pd.DataFrame() # Initialize an empty DataFrame
try:
    # This block will try to load your real 170,000+ song file
    song_df = pd.read_csv('data.csv')
    
    # Using your file's actual column names
    required_cols = ['artists', 'name', 'valence']
    missing_cols = [col for col in required_cols if col not in song_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in data.csv: {', '.join(missing_cols)}. "
                         "Please ensure 'artists', 'name', and 'valence' columns exist.")

    # Clean up the 'artists' column for better display
    song_df['artists'] = song_df['artists'].str.replace(r"[\[\]']", "", regex=True)

    song_df['valence'] = pd.to_numeric(song_df['valence'], errors='coerce')
    song_df.dropna(subset=['valence'], inplace=True)
    print("data.csv loaded successfully.")
    print(f"Number of songs loaded: {len(song_df)}") # Should be a large number

except FileNotFoundError:
    print("Error: data.csv not found. Please place your song dataset in the same directory.")
    print("Creating a dummy data.csv for demonstration purposes.")
    
    # Dummy data now uses the correct column names 'artists' and 'name'
    dummy_data = {
        'artists': ['Pharrell Williams', 'Adele', 'Ludovico Einaudi', 'Bobby McFerrin', 'Radiohead'],
        'name': ['Happy', 'Someone Like You', 'Nuvole Bianche', 'Don\'t Worry Be Happy', 'Creep'],
        'valence': [0.95, 0.1, 0.45, 0.88, 0.05]
    }
    song_df = pd.DataFrame(dummy_data)
    song_df.to_csv('data.csv', index=False)
    print("Dummy data.csv created. Please replace it with your actual dataset.")
    print(f"Number of songs loaded: {len(song_df)}")
    
except ValueError as ve:
    print(f"Data loading error: {ve}")
    print("Please fix your data.csv or ensure it has the correct columns.")
    song_df = pd.DataFrame() # Set to empty
except Exception as e:
    print(f"An unexpected error occurred loading or processing data.csv: {e}")
    traceback.print_exc()
    song_df = pd.DataFrame() # Set to empty

# --- 2. Text Sentiment Analysis (VADER) ---
def analyze_text_sentiment(text):
    """
    Analyzes text sentiment and returns a label and a compound score.
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    if compound_score <= -0.6:
        sentiment_label = "Very Sad 😭"
    elif compound_score <= -0.2:
        sentiment_label = "Sad 😢"
    elif compound_score < 0.2:
        sentiment_label = "Neutral 😐"
    elif compound_score < 0.6:
        sentiment_label = "Happy 🙂"
    else:
        sentiment_label = "Very Happy 😄"

    return sentiment_label, compound_score

# --- 3. Song Recommendation (Based on Text Sentiment) ---
def recommend_songs_from_text(text_score, num_recommendations=2):
    """
    Recommends songs from the DataFrame based on the text sentiment score.
    """
    global song_df
    if song_df.empty:
        print("No songs available in the dataset for recommendation.")
        return []

    # Map text sentiment score (-1 to 1) to valence (0 to 1)
    target_valence = (text_score + 1) / 2

    # Find the songs with the closest valence to our target
    song_df['valence_diff'] = abs(song_df['valence'] - target_valence)
    
    # Sort by the difference to get the closest matches
    closest_songs = song_df.sort_values(by='valence_diff').head(num_recommendations * 5)
    
    # Get unique songs from the closest matches
    unique_songs = closest_songs.drop_duplicates(subset=['name']).head(num_recommendations)

    # Using correct column names 'artists' and 'name'
    return unique_songs[['artists', 'name']].to_dict('records')


# --- Main Program Loop ---
def run_moodmate():
    """
    Main function to run the sentiment analysis and recommendation.
    """
    global song_df
    if song_df.empty:
        print("\nError: Song dataset could not be loaded or is empty. Cannot proceed.")
        return

    print("\n" + "="*50)
    print("🎵 MoodMate: Music Recommendation System")
    print("="*50 + "\n")

    # Get text input
    user_text = input("Tell me about your mood or what's on your mind: ")
    text_label, text_score = analyze_text_sentiment(user_text)

    # Recommend songs based on text score
    recommended_songs_list = recommend_songs_from_text(text_score, num_recommendations=2)

    # --- Output Results ---
    print("\n" + "-"*40)
    print(f"🎵 Your Sentiment: {text_label} ({text_score:.2f})")
    print("-"*40)

    if recommended_songs_list:
        print("\n✨ Song Suggestion(s) ✨")
        for song in recommended_songs_list:
            # Using .get() with correct column names 'artists' and 'name'
            artist = song.get('artists', 'Unknown Artist')
            track_name = song.get('name', 'Unknown Song')
            print(f"\"{track_name}\" by {artist}")
    else:
        print("\n😔 No songs could be recommended from the available dataset.")

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_moodmate()
'''






import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import torch
from transformers import pipeline
import os
import random
import traceback # Import traceback for detailed error reporting

# Ensure NLTK VADER lexicon is downloaded (run this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading NLTK 'vader_lexicon' for sentiment analysis...")
    nltk.download('vader_lexicon')
    print("NLTK 'vader_lexicon' downloaded.")
except Exception as e:
    print(f"Error checking/downloading NLTK 'vader_lexicon': {e}")
    print("VADER sentiment analysis might not function correctly.")


# --- 1. Load Local Datasets ---
song_df = pd.DataFrame() # Initialize an empty DataFrame
try:
    # --- THIS BLOCK WILL NOW RUN ---
    # (As long as data.csv is in the same folder)
    song_df = pd.read_csv('data.csv')
    
    # Using your file's actual column names
    required_cols = ['artists', 'name', 'valence']
    missing_cols = [col for col in required_cols if col not in song_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in data.csv: {', '.join(missing_cols)}. "
                         "Please ensure 'artists', 'name', and 'valence' columns exist.")

    # Clean up the 'artists' column for better display
    song_df['artists'] = song_df['artists'].str.replace(r"[\[\]']", "", regex=True)

    song_df['valence'] = pd.to_numeric(song_df['valence'], errors='coerce')
    song_df.dropna(subset=['valence'], inplace=True)
    print("data.csv loaded successfully.")
    print(f"Number of songs loaded: {len(song_df)}") # Should be a large number

except FileNotFoundError:
    print("Error: data.csv not found. Please place your song dataset in the same directory.")
    print("Creating a dummy data.csv for demonstration purposes.")
    
    # Dummy data now uses the correct column names 'artists' and 'name'
    dummy_data = {
        'artists': ['Pharrell Williams', 'Adele', 'Ludovico Einaudi', 'Bobby McFerrin', 'Radiohead', 'Daft Punk', 'Nine Inch Nails', 'Gary Jules', 'Billie Eilish', 'Louis Armstrong'],
        'name': ['Happy', 'Someone Like You', 'Nuvole Bianche', 'Don\'t Worry Be Happy', 'Creep', 'Around the World', 'Hurt', 'Mad World', 'Bad Guy', 'What a Wonderful World'],
        'valence': [0.95, 0.1, 0.45, 0.88, 0.05, 0.7, 0.02, 0.15, 0.18, 0.98]
    }
    song_df = pd.DataFrame(dummy_data)
    song_df.to_csv('data.csv', index=False)
    print("Dummy data.csv created. Please replace it with your actual dataset.")
    print(f"Number of songs loaded: {len(song_df)}")
    
except ValueError as ve:
    print(f"Data loading error: {ve}")
    print("Please fix your data.csv or ensure it has the correct columns.")
    song_df = pd.DataFrame() # Set to empty
except Exception as e:
    print(f"An unexpected error occurred loading or processing data.csv: {e}")
    traceback.print_exc()
    song_df = pd.DataFrame() # Set to empty

# --- 2. Text Sentiment Analysis (VADER) ---
def analyze_text_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    if compound_score <= -0.6:
        sentiment_label = "Very Sad 😭"
    elif compound_score <= -0.2:
        sentiment_label = "Sad 😢"
    elif compound_score < 0.2:
        sentiment_label = "Neutral 😐"
    elif compound_score < 0.6:
        sentiment_label = "Happy 🙂"
    else:
        sentiment_label = "Very Happy 😄"

    return sentiment_label, compound_score

# --- 3. Image Sentiment Analysis (Hugging Face Transformers) ---
print("Loading image sentiment analysis model (this may take a moment)...")
image_sentiment_pipeline = None # Initialize to None
try:
    
    # --- THIS IS THE FIXED LINE ---
    # Using a new, working model instead of the one that was deleted.
    image_sentiment_pipeline = pipeline("image-classification", model="ehcalabres/face-emotion-recognition", device="cpu")
    
    print("Image sentiment model loaded successfully.")
    
except Exception as e:
    print(f"\n--- DETAILED ERROR LOADING IMAGE MODEL ---")
    traceback.print_exc()
    print(f"Error message: {e}")
    print("------------------------------------------")
    print("Image sentiment will default to neutral due to model loading failure.")
    print("Possible reasons: No internet, firewall, or the model is unavailable.")

def analyze_image_sentiment(image_path):
    if not image_sentiment_pipeline:
        return "Neutral 😐 (Image model not loaded)", 0.0

    try:
        img = Image.open(image_path)
        predictions = image_sentiment_pipeline(img)

        top_prediction = predictions[0] # Get the most confident prediction
        label = top_prediction['label'].lower()
        score = top_prediction['score']

        sentiment_label = "Neutral 😐"
        unified_score = 0.0

        # Mapping emotion labels to your unified sentiment score range (-1 to 1)
        # This new model uses the same labels (anger, sadness, etc.)
        if label == "anger" or label == "disgust":
            sentiment_label = "Negative 😠 (Angry/Disgust)"
            unified_score = -score 
        elif label == "fear":
            sentiment_label = "Negative 😱 (Fear)"
            unified_score = -score * 0.8
        elif label == "sadness":
            sentiment_label = "Negative 😢 (Sad)"
            unified_score = -score * 0.6
        elif label == "happiness":
            sentiment_label = "Positive 😄 (Happy)"
            unified_score = score * 0.8
        elif label == "surprise":
            sentiment_label = "Neutral 😮 (Surprise)"
            unified_score = 0.0
        elif label == "neutral":
            sentiment_label = "Neutral 😐"
            unified_score = 0.0
        else:
            sentiment_label = f"Unknown Emotion ({label}) 😐"
            unified_score = 0.0

        return sentiment_label, unified_score

    except FileNotFoundError:
        return "Image not found", 0.0
    except Exception as e:
        print(f"\n--- ERROR PROCESSING IMAGE: {e} ---")
        traceback.print_exc()
        return f"Error processing image", 0.0

# --- 4. Unified Sentiment Scoring ---
def combine_sentiments(text_score, image_score):
    # Simple average.
    combined_compound_score = (text_score + image_score) / 2

    if combined_compound_score <= -0.6:
        combined_label = "Very Sad 😭"
    elif combined_compound_score <= -0.2:
        combined_label = "Sad 😢"
    elif combined_compound_score < 0.2:
        combined_label = "Neutral 😐"
    elif combined_compound_score < 0.6:
        combined_label = "Happy 🙂"
    else:
        combined_label = "Very Happy 😄"

    return combined_label, combined_compound_score

# --- 5. Song Recommendation ---
def recommend_songs(combined_score, num_recommendations=2):
    global song_df
    if song_df.empty:
        print("No songs available in the dataset for recommendation.")
        return []

    # Map combined sentiment score (-1 to 1) to valence (0 to 1)
    target_valence = (combined_score + 1) / 2

    song_df['valence_diff'] = abs(song_df['valence'] - target_valence)
    
    # Use sort_values to find closest matches
    closest_songs = song_df.sort_values(by='valence_diff').head(num_recommendations * 10)
    
    # Drop duplicates based on name to get unique tracks
    unique_songs = closest_songs.drop_duplicates(subset=['name']).head(num_recommendations)

    if len(unique_songs) < num_recommendations:
        # If we still don't have enough, fill with random songs
        needed = num_recommendations - len(unique_songs)
        remaining_pool = song_df.drop(unique_songs.index)
        if not remaining_pool.empty and len(remaining_pool) >= needed:
            fill_songs = remaining_pool.sample(n=needed, random_state=42)
            unique_songs = pd.concat([unique_songs, fill_songs])
        elif not remaining_pool.empty:
            unique_songs = pd.concat([unique_songs, remaining_pool.sample(n=len(remaining_pool), random_state=42)])

    # Return the correct column names 'artists' and 'name'
    return unique_songs[['artists', 'name']].to_dict('records')


# --- Main Program Loop ---
def run_moodmate():
    global song_df
    if song_df.empty:
        print("\nError: Song dataset could not be loaded or is empty. Cannot proceed.")
        return

    print("\n" + "="*50)
    print("🎵 MoodMate: Emotion Detection and Music Recommendation System")
    print("="*50 + "\n")

    # Get text input
    user_text = input("Tell me about your mood or what's on your mind: ")
    text_label, text_score = analyze_text_sentiment(user_text)

    # Get image input
    image_path = input("Enter the path to an image file (e.g., 'my_mood.jpg'): ")
    image_label = "Image analysis skipped"
    image_score = 0.0 

    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at '{image_path}'. Skipping image analysis.")
    else:
        current_image_label, current_image_score = analyze_image_sentiment(image_path)
        if "Error processing image" in current_image_label:
            print(f"Image processing error. Defaulting image sentiment to neutral.")
            image_label = "Error (Defaulted to Neutral)"
            image_score = 0.0
        elif "Image model not loaded" in current_image_label:
            print(f"Warning: {current_image_label}. Defaulting image sentiment to neutral.")
            image_label = "Model Error (Defaulted to Neutral)"
            image_score = 0.0
        else:
            image_label = current_image_label
            image_score = current_image_score

    # Combine sentiments
    combined_label, combined_score = combine_sentiments(text_score, image_score)

    # Recommend songs
    recommended_songs_list = recommend_songs(combined_score, num_recommendations=2)

    # --- Output Results ---
    print("\n" + "-"*40)
    print(f"🎵 Combined Sentiment: {combined_label} ({combined_score:.2f})")
    print(f"🧠 Text Sentiment: {text_label} ({text_score:.2f})")
    print(f"🖼️ Image Sentiment: {image_label} ({image_score:.2f})")
    print("-"*40)

    if recommended_songs_list:
        print("\n✨ Song Suggestion(s) ✨")
        for song in recommended_songs_list:
            # Using .get() with the correct column names
            artist = song.get('artists', 'Unknown Artist')
            track_name = song.get('name', 'Unknown Song')
            print(f"\"{track_name}\" by {artist}")
    else:
        print("\n😔 No songs could be recommended from the available dataset.")

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_moodmate()
