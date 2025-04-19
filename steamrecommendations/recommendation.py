# filepath: c:\Users\aobai\Documents\Programming Stuff\Steam Game Recommendation Website\steamrecommendation\steamrecommendations\recommendations.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from django.core.cache import cache
import os
import joblib # For saving/loading the model

DATA_DIR = './data'
MODEL_DIR = './model_cache' # Directory to save/load the model
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_model.joblib')
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix.joblib')
GAMES_DF_PATH = os.path.join(MODEL_DIR, 'games_df.joblib')
SAMPLE_DF_PATH = os.path.join(MODEL_DIR, 'sample_df.joblib')

def load_data():
    """Loads the necessary dataframes."""
    try:
        df_games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'), dtype={'app_id': 'int32', 'title': 'string'})
        df_games_meta = pd.read_json(os.path.join(DATA_DIR, 'games_metadata.json'), lines=True, orient="records")
        # Ensure app_id is integer for merging
        df_games_meta['app_id'] = pd.to_numeric(df_games_meta['app_id'], errors='coerce').fillna(0).astype(int)
        df_games['app_id'] = pd.to_numeric(df_games['app_id'], errors='coerce').fillna(0).astype(int)
        return df_games, df_games_meta
    except FileNotFoundError:
        print("Error: Data files not found in ./data directory.")
        return None, None

def train_and_save_model(sample_size=20000, force_retrain=False):
    """Trains the TF-IDF model and saves it along with the data."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not force_retrain and os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(GAMES_DF_PATH) and os.path.exists(SAMPLE_DF_PATH):
        print("Model and data already exist. Skipping training.")
        return

    print("Loading data for training...")
    df_games, df_games_meta = load_data()
    if df_games is None or df_games_meta is None:
        return

    print(f"Original metadata size: {len(df_games_meta)}")
    df_games_meta.dropna(subset=['app_id'], inplace=True) # Ensure app_id is not NaN
    df_games_meta['app_id'] = df_games_meta['app_id'].astype(int)

    # Ensure sample size is not larger than available data
    actual_sample_size = min(sample_size, df_games_meta.shape[0])
    if actual_sample_size < sample_size:
         print(f"Warning: Sample size reduced to {actual_sample_size} due to available data.")

    if actual_sample_size == 0:
        print("Error: No data available for sampling after cleaning.")
        return

    print(f"Sampling {actual_sample_size} games...")
    np.random.seed(42)
    sample_indices = np.random.choice(df_games_meta.index, actual_sample_size, replace=False)
    df_sample = df_games_meta.loc[sample_indices].reset_index(drop=True)

    print("Preprocessing text data...")
    df_sample['tags_str'] = df_sample['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df_sample['combined_text'] = df_sample['description'].fillna('') + ' ' + df_sample['tags_str']

    print("Training TF-IDF model...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_sample['combined_text'])

    print("Saving model and data...")
    joblib.dump(tfidf, TFIDF_MODEL_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    joblib.dump(df_games, GAMES_DF_PATH) # Save the games df used
    joblib.dump(df_sample[['app_id', 'description', 'tags_str']], SAMPLE_DF_PATH) # Save only necessary columns of sample

    print("Model training and saving complete.")


def load_model_and_data():
    """Loads the pre-trained model and associated data."""
    cache_key_model = "tfidf_model"
    cache_key_matrix = "tfidf_matrix"
    cache_key_games = "tfidf_games_df"
    cache_key_sample = "tfidf_sample_df"

    model = cache.get(cache_key_model)
    matrix = cache.get(cache_key_matrix)
    games_df = cache.get(cache_key_games)
    sample_df = cache.get(cache_key_sample)

    if all([model, matrix is not None, games_df is not None, sample_df is not None]):
        return model, matrix, games_df, sample_df

    if not os.path.exists(TFIDF_MODEL_PATH):
         print("Model files not found. Training model...")
         train_and_save_model() # Train if files don't exist

    try:
        print("Loading model and data from disk...")
        model = joblib.load(TFIDF_MODEL_PATH)
        matrix = joblib.load(TFIDF_MATRIX_PATH)
        games_df = joblib.load(GAMES_DF_PATH)
        sample_df = joblib.load(SAMPLE_DF_PATH)

        # Cache the loaded data
        cache.set(cache_key_model, model, timeout=None) # Cache indefinitely
        cache.set(cache_key_matrix, matrix, timeout=None)
        cache.set(cache_key_games, games_df, timeout=None)
        cache.set(cache_key_sample, sample_df, timeout=None)

        return model, matrix, games_df, sample_df
    except FileNotFoundError:
        print("Error: Could not load model files.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred loading model/data: {e}")
        return None, None, None, None


def get_content_recommendations(app_id, n=10):
    """Gets content-based recommendations for a given app_id."""
    tfidf, tfidf_matrix, df_games, df_sample = load_model_and_data()

    if tfidf is None or tfidf_matrix is None or df_games is None or df_sample is None:
        return pd.DataFrame(), None # Return empty DataFrame and None for source game

    # Find the index in the sample dataframe
    try:
        # Ensure app_id is integer
        app_id = int(app_id)
        idx = df_sample[df_sample['app_id'] == app_id].index[0]
        source_game_title = df_games[df_games['app_id'] == app_id]['title'].iloc[0] if not df_games[df_games['app_id'] == app_id].empty else f"Game ID {app_id}"

    except (IndexError, ValueError):
        print(f"Game with app_id '{app_id}' not found in the recommendation sample dataset.")
        return pd.DataFrame(), None

    # Compute similarity
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1] # Exclude self

    # Get recommended game indices and app_ids
    game_indices = [i[0] for i in sim_scores]
    recommended_app_ids = df_sample.iloc[game_indices]['app_id'].values

    # Fetch details from the original df_games
    recommended_games = df_games[df_games['app_id'].isin(recommended_app_ids)].copy()

    # Add similarity scores (optional)
    sim_dict = {df_sample.iloc[i]['app_id']: score for i, score in sim_scores}
    recommended_games['similarity'] = recommended_games['app_id'].map(sim_dict)
    recommended_games = recommended_games.sort_values('similarity', ascending=False)


    # Merge with full game details if needed (example: price, rating)
    full_details = df_games[['app_id', 'title', 'date_release', 'rating', 'price_final']].copy()
    recommended_games = pd.merge(
        recommended_games[['app_id', 'title', 'similarity']],
        full_details,
        on='app_id',
        how='left'
    ).rename(columns={'title_y': 'title'}).drop('title_x', axis=1, errors='ignore')


    return recommended_games, source_game_title

# You might want a management command to run this initially
# Example: python manage.py train_rec_model
# train_and_save_model() # Call this once manually or via management command