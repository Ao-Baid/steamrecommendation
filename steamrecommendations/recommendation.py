# filepath: c:\Users\aobai\Documents\Programming Stuff\Steam Game Recommendation Website\steamrecommendation\steamrecommendations\recommendations.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from django.core.cache import cache
import os
import joblib # For saving/loading the model
import requests
import json
from bs4 import BeautifulSoup

DATA_DIR = './data'
MODEL_DIR = './model_cache' # Directory to save/load the model
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_model_full.joblib') # Changed filename
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix_full.joblib') # Changed filename
GAMES_DF_PATH = os.path.join(MODEL_DIR, 'games_df_full.joblib') # Changed filename
META_DF_PATH = os.path.join(MODEL_DIR, 'meta_df_full.joblib') # Changed filename and variable

def load_data():
    """Loads the necessary dataframes."""
    try:
        df_games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'), dtype={'app_id': 'int32', 'title': 'string'})
        df_games_meta = pd.read_json(os.path.join(DATA_DIR, 'games_metadata.json'), lines=True, orient="records")
        # Ensure app_id is integer for merging
        df_games_meta['app_id'] = pd.to_numeric(df_games_meta['app_id'], errors='coerce')
        df_games['app_id'] = pd.to_numeric(df_games['app_id'], errors='coerce')
        # Drop rows where app_id could not be converted in either dataframe
        df_games_meta.dropna(subset=['app_id'], inplace=True)
        df_games.dropna(subset=['app_id'], inplace=True)
        df_games_meta['app_id'] = df_games_meta['app_id'].astype(int)
        df_games['app_id'] = df_games['app_id'].astype(int)
        return df_games, df_games_meta
    except FileNotFoundError:
        print("Error: Data files not found in ./data directory.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Removed sample_size parameter as we use the full dataset now
def train_and_save_model(force_retrain=False):
    """Trains the TF-IDF model on the FULL dataset and saves it."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if full model files exist
    if not force_retrain and os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(GAMES_DF_PATH) and os.path.exists(META_DF_PATH):
        print("Full model and data already exist. Skipping training.")
        return

    print("Loading data for full training...")
    df_games, df_games_meta = load_data()
    if df_games is None or df_games_meta is None:
        print("Failed to load data for training.")
        return

    print(f"Using full metadata size: {len(df_games_meta)}")
    # Basic cleaning (already done in load_data, but ensure no NaN app_id)
    df_games_meta.dropna(subset=['app_id'], inplace=True)
    df_games_meta['app_id'] = df_games_meta['app_id'].astype(int)

    if df_games_meta.empty:
        print("Error: No data available in metadata after cleaning.")
        return

    # Use the entire df_games_meta dataframe
    df_processed = df_games_meta.copy()

    print("Preprocessing text data for full dataset...")
    # Handle potential non-list 'tags' entries more robustly
    df_processed['tags_str'] = df_processed['tags'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else (str(x) if pd.notna(x) else '')
    )
    df_processed['combined_text'] = df_processed['description'].fillna('') + ' ' + df_processed['tags_str']

    print("Training TF-IDF model on full dataset...")
    # Consider adjusting max_features if memory becomes an issue
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000) # Increased max_features
    tfidf_matrix = tfidf.fit_transform(df_processed['combined_text'])

    print("Saving full model and data...")
    joblib.dump(tfidf, TFIDF_MODEL_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    # Save the full df_games and the processed metadata df (only necessary columns)
    joblib.dump(df_games, GAMES_DF_PATH)
    joblib.dump(df_processed[['app_id', 'description', 'tags_str']], META_DF_PATH)

    print("Full model training and saving complete.")


def load_model_and_data():
    """Loads the pre-trained FULL model and associated data."""
    # Update cache keys for full model
    cache_key_model = "tfidf_model_full"
    cache_key_matrix = "tfidf_matrix_full"
    cache_key_games = "tfidf_games_df_full"
    cache_key_meta = "tfidf_meta_df_full" # Changed from sample

    model = cache.get(cache_key_model)
    matrix = cache.get(cache_key_matrix)
    games_df = cache.get(cache_key_games)
    meta_df = cache.get(cache_key_meta) # Changed from sample_df

    # Check if all components are in cache
    if all([model, matrix is not None, games_df is not None, meta_df is not None]):
        print("Loading full model and data from cache...")
        return model, matrix, games_df, meta_df # Return meta_df

    # Check if model files exist on disk
    if not os.path.exists(TFIDF_MODEL_PATH) or not os.path.exists(TFIDF_MATRIX_PATH) or not os.path.exists(GAMES_DF_PATH) or not os.path.exists(META_DF_PATH):
         print("Full model files not found. Training full model...")
         train_and_save_model() # Train if files don't exist

    try:
        print("Loading full model and data from disk...")
        model = joblib.load(TFIDF_MODEL_PATH)
        matrix = joblib.load(TFIDF_MATRIX_PATH)
        games_df = joblib.load(GAMES_DF_PATH)
        meta_df = joblib.load(META_DF_PATH) # Load meta_df

        # Cache the loaded data
        print("Caching full model and data...")
        cache.set(cache_key_model, model, timeout=None) # Cache indefinitely
        cache.set(cache_key_matrix, matrix, timeout=None)
        cache.set(cache_key_games, games_df, timeout=None)
        cache.set(cache_key_meta, meta_df, timeout=None) # Cache meta_df

        return model, matrix, games_df, meta_df # Return meta_df
    except FileNotFoundError:
        print("Error: Could not load full model files from disk.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred loading full model/data: {e}")
        return None, None, None, None

def _clean_html(raw_html):
    """Helper function to clean HTML content."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        # Get text, replace consecutive whitespace with a single space, strip ends
        text = ' '.join(soup.get_text(separator=' ', strip=True).split())
        return text
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return "" # Return empty string on error

def get_steam_app_details(app_id):
    """Fetches app details from Steam Storefront API, cleans descriptions, and extracts key text."""
    try:
        print(f"\n==== FETCHING BASIC DATA FOR APP ID: {app_id} ====")
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        app_id_str = str(app_id)
        if app_id_str in data and data[app_id_str]['success']:
            app_data = data[app_id_str]['data']

            # Extract and clean data
            name = app_data.get('name', '')
            detailed_desc_cleaned = _clean_html(app_data.get('detailed_description', ''))
            short_desc_cleaned = _clean_html(app_data.get('short_description', ''))

            # Extract genres
            genres_list = [
                genre['description'] for genre in app_data.get('genres', [])
                if isinstance(genre, dict) and 'description' in genre
            ]
            genres_str = ' '.join(genres_list)

            # Extract categories (treat as tags)
            categories_list = [
                cat['description'] for cat in app_data.get('categories', [])
                if isinstance(cat, dict) and 'description' in cat
            ]
            categories_str = ' '.join(categories_list)

            # Combine relevant text fields for TF-IDF
            # Prioritize detailed description, add short description, name, genres, categories
            combined_text = f"{name} {detailed_desc_cleaned} {short_desc_cleaned} {genres_str} {categories_str}".strip()
            # Remove extra whitespace
            combined_text = ' '.join(combined_text.split())

            source_game_title = name if name else f"Game ID {app_id}"

            # Print out what we got
            print(f"✓ STEAM DATA FETCHED: {source_game_title}")
            print(f"✓ GENRES: {', '.join(genres_list)}")
            print(f"✓ CATEGORIES: {', '.join(categories_list)}")
            print(f"✓ DETAILED DESC LEN (Cleaned): {len(detailed_desc_cleaned)}")
            print(f"✓ SHORT DESC LEN (Cleaned): {len(short_desc_cleaned)}")
            print(f"✓ COMBINED TEXT LEN: {len(combined_text)}")
            print(f"✓ DETAILED DESC PREVIEW (Cleaned): {detailed_desc_cleaned[:150]}...")
            print(f"==== END FETCHED DATA ====\n")

            # Return combined text for TF-IDF and the title
            return combined_text, source_game_title
        else:
            api_status = data.get(app_id_str, {}).get('success', 'Unknown')
            print(f"✗ STEAM API REQUEST FAILED for {app_id}: Success status: {api_status}")
            print(f"==== END FETCHED DATA ====\n")
            return None, f"Game ID {app_id}"
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR FETCHING STEAM API DATA for {app_id}: {e}")
        print(f"==== END FETCHED DATA ====\n")
        return None, f"Game ID {app_id}"
    except json.JSONDecodeError as e:
        print(f"✗ ERROR DECODING JSON FROM STEAM API for {app_id}: {e}")
        print(f"==== END FETCHED DATA ====\n")
        return None, f"Game ID {app_id}"
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR FETCHING STEAM DETAILS for {app_id}: {e}")
        print(f"==== END FETCHED DATA ====\n")
        return None, f"Game ID {app_id}"


def get_tag_based_recommendations(app_id_int, df_meta, df_games, source_tags=None, n=10):
    """Fallback recommendation function that uses tags directly instead of TF-IDF."""
    print(f"Using tag-based fallback for {app_id_int}")
    
    # If source_tags was provided (from API), use those
    if source_tags is None:
        # Try to find the source game in df_meta
        source_game = df_meta[df_meta['app_id'] == app_id_int]
        if source_game.empty:
            print(f"Game {app_id_int} not found in metadata for tag-based recommendations")
            return pd.DataFrame()
            
        # Get the source game's tags
        source_tags_str = source_game['tags_str'].iloc[0] if 'tags_str' in source_game.columns else ""
        if not source_tags_str or pd.isna(source_tags_str):
            print(f"Game {app_id_int} has no tags for tag-based recommendations")
            return pd.DataFrame()
            
        source_tags = set(source_tags_str.lower().split())
    
    if len(source_tags) == 0:
        print(f"Game {app_id_int} has empty tags after processing")
        return pd.DataFrame()
    
    print(f"Source tags for {app_id_int}: {source_tags}")
    
    # Calculate tag overlap for all games
    tag_similarity = []
    for idx, row in df_meta.iterrows():
        if row['app_id'] == app_id_int:  # Skip the source game
            continue
            
        game_tags_str = row['tags_str'] if 'tags_str' in row and pd.notna(row['tags_str']) else ""
        game_tags = set(game_tags_str.lower().split())
        
        if len(game_tags) == 0:
            continue
            
        # Calculate Jaccard similarity: size of intersection / size of union
        intersection = len(source_tags.intersection(game_tags))
        union = len(source_tags.union(game_tags))
        
        if union > 0 and intersection > 0:  # Must have at least some overlap
            similarity = intersection / union
            tag_similarity.append((idx, similarity, row['app_id']))
    
    # Sort by similarity
    tag_similarity.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_similar = tag_similarity[:n]
    
    if not top_similar:
        print(f"No tag-based recommendations found for {app_id_int}")
        return pd.DataFrame()
    
    # Get the app_ids
    similar_app_ids = [x[2] for x in top_similar]
    
    # Get full game details
    recommended_games = df_games[df_games['app_id'].isin(similar_app_ids)].copy()
    
    if recommended_games.empty:
        print(f"Could not find game details for tag-based recommendations for {app_id_int}")
        return pd.DataFrame()
        
    # Add similarity scores
    similarity_dict = {app_id: sim for _, sim, app_id in top_similar}
    recommended_games['similarity'] = recommended_games['app_id'].map(similarity_dict)
    recommended_games['similarity'].fillna(0, inplace=True)
    recommended_games = recommended_games.sort_values('similarity', ascending=False)
    
    print(f"Found {len(recommended_games)} tag-based recommendations for {app_id_int}")
    return recommended_games


def get_enhanced_game_data(app_id_int):
    """Fetch enhanced game data from Steam API, clean descriptions, and extract structured information."""
    try:
        print(f"\n==== FETCHING ENHANCED DATA FOR APP ID: {app_id_int} ====")
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id_int}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        app_id_str = str(app_id_int)
        if app_id_str in data and data[app_id_str]['success']:
            app_data = data[app_id_str]['data']

            # Extract and clean data
            name = app_data.get('name', '')
            detailed_desc_cleaned = _clean_html(app_data.get('detailed_description', ''))
            short_desc_cleaned = _clean_html(app_data.get('short_description', ''))

            # Extract genres
            genres_list = [
                genre['description'] for genre in app_data.get('genres', [])
                if isinstance(genre, dict) and 'description' in genre
            ]

            # Extract categories
            categories_list = [
                cat['description'] for cat in app_data.get('categories', [])
                if isinstance(cat, dict) and 'description' in cat
            ]

            # Combine genres and categories into a single tag list/string
            all_tags_list = genres_list + categories_list
            all_tags_str = ' '.join(all_tags_list)

            # Combine relevant text fields for TF-IDF
            combined_text = f"{name} {detailed_desc_cleaned} {short_desc_cleaned} {all_tags_str}".strip()
            combined_text = ' '.join(combined_text.split()) # Remove extra whitespace

            # Print out what we got
            print(f"✓ STEAM DATA FETCHED: {name}")
            print(f"✓ GENRES: {', '.join(genres_list)}")
            print(f"✓ CATEGORIES: {', '.join(categories_list)}")
            print(f"✓ DETAILED DESC LEN (Cleaned): {len(detailed_desc_cleaned)}")
            print(f"✓ SHORT DESC LEN (Cleaned): {len(short_desc_cleaned)}")
            print(f"✓ COMBINED TEXT LEN: {len(combined_text)}")
            print(f"✓ DETAILED DESC PREVIEW (Cleaned): {detailed_desc_cleaned[:150]}...")
            print(f"==== END FETCHED DATA ====\n")

            # Return structured data with cleaned descriptions and combined tags
            return {
                'title': name,
                'description': detailed_desc_cleaned, # Return cleaned detailed description
                'short_description': short_desc_cleaned, # Return cleaned short description
                'tags': all_tags_list, # Combined list of genres and categories
                'tags_str': all_tags_str, # Combined string of genres and categories
                'combined_text': combined_text # Text for TF-IDF
            }
        else:
            api_status = data.get(app_id_str, {}).get('success', 'Unknown')
            print(f"✗ STEAM API REQUEST FAILED for {app_id_int}: Success status: {api_status}")
            print(f"==== END FETCHED DATA ====\n")
    except Exception as e:
        print(f"✗ ERROR FETCHING ENHANCED DATA for {app_id_int}: {e}")
        print(f"==== END FETCHED DATA ====\n")

    return None


def get_content_recommendations(app_id, n=10, min_similarity=0.08):
    """Gets content-based recommendations for a given app_id using the FULL dataset."""
    tfidf, tfidf_matrix, df_games, df_meta = load_model_and_data()

    if tfidf is None or tfidf_matrix is None or df_games is None or df_meta is None:
        print(f"Failed to load model/data for recommendations for app_id {app_id}")
        _, source_game_title = get_steam_app_details(app_id)
        return pd.DataFrame(), source_game_title if source_game_title else f"Game ID {app_id}"

    source_game_title = f"Game ID {app_id}" # Default title
    app_id_int = None
    enhanced_data = None
    try:
        # Ensure app_id is integer
        app_id_int = int(app_id)
        
        # Find the index in the metadata dataframe
        idx_series = df_meta[df_meta['app_id'] == app_id_int].index
        if not idx_series.empty:
            idx = idx_series[0]
            
            # Check if this game has good metadata (tags and description)
            has_good_metadata = True
            if ('tags_str' not in df_meta.columns or 
                pd.isna(df_meta.loc[idx, 'tags_str']) or 
                df_meta.loc[idx, 'tags_str'].strip() == '' or
                'description' not in df_meta.columns or
                pd.isna(df_meta.loc[idx, 'description']) or
                df_meta.loc[idx, 'description'].strip() == ''):
                
                has_good_metadata = False
                print(f"Game {app_id_int} has incomplete metadata. Attempting to enhance...")
                
                # Try to fetch enhanced data
                enhanced_data = get_enhanced_game_data(app_id_int)
                if enhanced_data and enhanced_data['tags']:
                    print(f"Successfully enhanced metadata for {app_id_int}")
                    
                    # Process with the enhanced data directly
                    # We'll transform this single enhanced text using the existing TF-IDF model
                    enhanced_text = enhanced_data['combined_text']
                    app_vector = tfidf.transform([enhanced_text])
                    cosine_sim = linear_kernel(app_vector, tfidf_matrix).flatten()
                    
                    # Update title from enhanced data
                    source_game_title = enhanced_data['title']
                else:
                    # If we couldn't enhance, continue with what we have
                    print(f"Could not enhance metadata for {app_id_int}, using existing data")
            
            # If we didn't use enhanced data, use the existing row from df_meta
            if not enhanced_data:
                # Get title from df_games if possible
                source_game_title_series = df_games[df_games['app_id'] == app_id_int]['title']
                if not source_game_title_series.empty:
                    source_game_title = source_game_title_series.iloc[0]
                else: # Fallback to API if not in df_games
                    _, api_title = get_steam_app_details(app_id_int)
                    if api_title: source_game_title = api_title

                # Compute similarity using existing matrix row
                print(f"Calculating similarity for existing game {app_id_int} ('{source_game_title}', index {idx})")
                cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        else:
            # Game not found in df_meta, trigger cold start logic below
            raise IndexError("Game not found in metadata")

    except (IndexError, ValueError, TypeError) as e:
        # --- Cold Start Logic ---
        print(f"Game with app_id '{app_id}' not found in the local metadata. Attempting cold start via Steam API... ({e})")
        if app_id_int is None: # Handle case where initial int conversion failed
            try: app_id_int = int(app_id)
            except (ValueError, TypeError):
                print(f"Invalid app_id format for cold start: {app_id}")
                return pd.DataFrame(), f"Invalid Game ID {app_id}"

        # Try to get enhanced data first (more complete)
        enhanced_data = get_enhanced_game_data(app_id_int)
        if enhanced_data:
            source_game_title = enhanced_data['title']
            app_text = enhanced_data['combined_text']
            source_tags = set(enhanced_data['tags_str'].lower().split())
        else:
            # Fall back to basic API call
            app_text, api_source_title = get_steam_app_details(app_id_int)
            source_game_title = api_source_title
            source_tags = None  # We don't have tags from the basic API call

        if app_text:
            try:
                # Transform the new game's text using the loaded TF-IDF model
                print(f"Transforming text for cold start game {app_id_int}")
                app_vector = tfidf.transform([app_text])
                # Compute similarity against the entire existing matrix
                print(f"Calculating similarity for cold start game {app_id_int}")
                cosine_sim = linear_kernel(app_vector, tfidf_matrix).flatten()
                print(f"Successfully generated recommendations for {app_id_int} via cold start.")
            except Exception as e:
                print(f"Error processing cold start vector for {app_id_int}: {e}")
                # Try tag-based as last resort if we have tags
                if source_tags and len(source_tags) > 0:
                    tag_recs = get_tag_based_recommendations(app_id_int, df_meta, df_games, source_tags, n)
                    if not tag_recs.empty:
                        return tag_recs, source_game_title
                return pd.DataFrame(), source_game_title
        else:
            print(f"Could not fetch data for cold start for app_id {app_id_int}.")
            return pd.DataFrame(), source_game_title
        # --- End Cold Start Logic ---

    # If cosine_sim was not calculated (e.g., cold start failed before calculation)
    if 'cosine_sim' not in locals():
        print(f"Cosine similarity calculation failed for {app_id}")
        return pd.DataFrame(), source_game_title

    # Get similarity scores (index, score)
    sim_scores = list(enumerate(cosine_sim))
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Print the highest similarity score for debugging
    if sim_scores:
        highest_similarity = sim_scores[0][1]
        print(f"Highest similarity score for {app_id_int} ('{source_game_title}'): {highest_similarity:.4f}")
    
    # Exclude self if the game was found in the original dataset ('idx' is defined)
    if 'idx' in locals() and idx is not None:
        sim_scores = [score for score in sim_scores if score[0] != idx]

    # Filter by minimum similarity threshold
    relevant_scores = [score for score in sim_scores if score[1] >= min_similarity]
    
    # If we have enough relevant recommendations, use those
    # Otherwise try tag-based recommendations if we have enhanced data with tags
    if len(relevant_scores) >= 3:
        print(f"Found {len(relevant_scores)} games with similarity >= {min_similarity}")
        sim_scores = relevant_scores[:n]
    else:
        print(f"WARNING: Only {len(relevant_scores)} games with similarity >= {min_similarity}")
        
        # Try tag-based approach if TF-IDF gave poor results
        if enhanced_data and enhanced_data['tags']:
            print(f"Trying tag-based recommendations for {app_id_int} with enhanced tags")
            source_tags = set(enhanced_data['tags_str'].lower().split())
            tag_recs = get_tag_based_recommendations(app_id_int, df_meta, df_games, source_tags, n)
            
            if not tag_recs.empty and len(tag_recs) >= 3:
                print(f"Using tag-based recommendations for {app_id_int}")
                return tag_recs, source_game_title
        
        # If tag-based also failed or wasn't attempted, use the best TF-IDF scores we have
        sim_scores = sim_scores[:n]

    # Get recommended game indices
    game_indices = [i[0] for i in sim_scores]
    # Ensure indices are within bounds of df_meta
    valid_indices = [idx for idx in game_indices if idx < len(df_meta)]
    if len(valid_indices) != len(game_indices):
        print(f"Warning: Some recommendation indices were out of bounds for df_meta (length {len(df_meta)})")
        game_indices = valid_indices

    if not game_indices:
        print(f"No valid recommendation indices found for {app_id}")
        return pd.DataFrame(), source_game_title

    recommended_app_ids = df_meta.iloc[game_indices]['app_id'].values
    recommended_games = df_games[df_games['app_id'].isin(recommended_app_ids)].copy()

    # Add similarity scores
    sim_dict_by_index = {i: score for i, score in sim_scores}
    index_to_appid = df_meta.iloc[game_indices]['app_id'].to_dict()
    sim_dict_by_appid = {app_id: sim_dict_by_index[index] for index, app_id in index_to_appid.items() if index in sim_dict_by_index}

    recommended_games['similarity'] = recommended_games['app_id'].map(sim_dict_by_appid)
    recommended_games['similarity'] = recommended_games['similarity'].fillna(0)
    recommended_games = recommended_games.sort_values('similarity', ascending=False)

    # Ensure essential columns exist
    for col in ['app_id', 'title', 'date_release', 'rating', 'price_final', 'similarity']:
        if col not in recommended_games.columns:
            recommended_games[col] = None if col != 'similarity' else 0

    print(f"Returning {len(recommended_games)} recommendations for {app_id} ('{source_game_title}')")
    return recommended_games, source_game_title

