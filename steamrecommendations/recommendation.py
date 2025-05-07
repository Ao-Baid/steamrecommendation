# filepath: c:\Users\aobai\Documents\Programming Stuff\Steam Game Recommendation Website\steamrecommendation\steamrecommendations\recommendations.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from django.core.cache import cache
from .tf_idf_training import train_and_save_model, _clean_html # Import the training function
import os
import joblib # For saving/loading the model
import requests
import json

DATA_DIR = './data'
MODEL_DIR = './model_cache' # Directory to save/load the model
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_model_full.joblib') # Changed filename
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix_full.joblib') # Changed filename
GAMES_DF_PATH = os.path.join(MODEL_DIR, 'games_df_full.joblib') # Changed filename
META_DF_PATH = os.path.join(MODEL_DIR, 'meta_df_full.joblib') # Changed filename and variable
ITEM_SIMILARITY_PATH = os.path.join(MODEL_DIR, 'item_similarity.joblib') # Path for item similarity matrix
APP_ID_MAP_PATH = os.path.join(MODEL_DIR, 'item_similarity_app_id_map.joblib') # Path for app ID mapping

# Cache for loaded similarity matrix
_item_similarity_cache = None
_df_games_cache = None  # Assuming you have a global df_games cache elsewhere


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


def get_content_recommendations(app_id, n=12, min_similarity=0.08):
    """Gets content-based recommendations for a given app_id using the FULL dataset."""
    tfidf, tfidf_matrix, df_games, df_meta = load_model_and_data()

    if tfidf is None or tfidf_matrix is None or df_games is None or df_meta is None:
        print(f"✗ Failed to load model/data for recommendations for app_id {app_id}")
        # Try to get title from API as fallback
        _, source_game_title = get_steam_app_details(app_id)
        return pd.DataFrame(), source_game_title if source_game_title else f"Game ID {app_id}"
    else:
        print("✓ Model and data loaded successfully.")

    source_game_title = f"Game ID {app_id}" # Default title
    app_id_int = None
    enhanced_data = None
    cosine_sim = None # Initialize cosine_sim

    try:
        # Ensure app_id is integer
        app_id_int = int(app_id)
        print(f"✓ Processing recommendations for App ID: {app_id_int}")

        # Find the index in the metadata dataframe
        idx_series = df_meta[df_meta['app_id'] == app_id_int].index
        if not idx_series.empty:
            idx = idx_series[0]
            print(f"✓ Game found in local metadata at index {idx}.")

            # Check if this game has good metadata (tags and description)
            has_good_metadata = True
            if ('tags_str' not in df_meta.columns or
                pd.isna(df_meta.loc[idx, 'tags_str']) or
                df_meta.loc[idx, 'tags_str'].strip() == '' or
                'description' not in df_meta.columns or
                pd.isna(df_meta.loc[idx, 'description']) or
                df_meta.loc[idx, 'description'].strip() == ''):

                has_good_metadata = False
                print(f"ℹ Game {app_id_int} has incomplete metadata. Attempting to enhance via Steam API...")

                # Try to fetch enhanced data
                enhanced_data = get_enhanced_game_data(app_id_int) # Uses the enhanced fetcher
                if enhanced_data and enhanced_data.get('combined_text'): # Check if we got useful text
                    print(f"✓ Successfully enhanced metadata for {app_id_int}")

                    # Process with the enhanced data directly
                    enhanced_text = enhanced_data['combined_text']
                    app_vector = tfidf.transform([enhanced_text])
                    print(f"✓ Transformed enhanced text for {app_id_int}")
                    cosine_sim = linear_kernel(app_vector, tfidf_matrix).flatten()
                    print(f"✓ Calculated similarity using enhanced data.")

                    # Update title from enhanced data
                    source_game_title = enhanced_data.get('title', f"Game ID {app_id_int}")
                else:
                    # If we couldn't enhance, continue with what we have
                    print(f"✗ Could not enhance metadata for {app_id_int}, using existing (potentially incomplete) data.")
                    # Fall through to use existing matrix row

            # If we didn't use enhanced data (either it was good initially or enhancement failed)
            if cosine_sim is None: # Check if cosine_sim wasn't calculated yet
                # Get title from df_games if possible
                source_game_title_series = df_games[df_games['app_id'] == app_id_int]['title']
                if not source_game_title_series.empty:
                    source_game_title = source_game_title_series.iloc[0]
                else: # Fallback to API if not in df_games (should be rare if in df_meta)
                    _, api_title = get_steam_app_details(app_id_int)
                    if api_title: source_game_title = api_title

                # Compute similarity using existing matrix row
                print(f"✓ Calculating similarity for existing game {app_id_int} ('{source_game_title}', index {idx}) using precomputed matrix.")
                cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        else:
            # Game not found in df_meta, trigger cold start logic below
            print(f"ℹ Game {app_id_int} not found in local metadata.")
            raise IndexError("Game not found in metadata") # Raise to enter cold start

    except (IndexError, ValueError, TypeError) as e:
        # --- Cold Start Logic ---
        print(f"ℹ Attempting cold start for App ID '{app_id}' via Steam API... (Triggered by: {e})")
        if app_id_int is None: # Handle case where initial int conversion failed
            try: app_id_int = int(app_id)
            except (ValueError, TypeError):
                print(f"✗ Invalid app_id format for cold start: {app_id}")
                return pd.DataFrame(), f"Invalid Game ID {app_id}"

        # Try to get enhanced data first (more complete)
        enhanced_data = get_enhanced_game_data(app_id_int)
        source_tags = None # Initialize source_tags

        if enhanced_data and enhanced_data.get('combined_text'):
            source_game_title = enhanced_data.get('title', f"Game ID {app_id_int}")
            app_text = enhanced_data['combined_text']
            source_tags = set(enhanced_data.get('tags_str', '').lower().split()) # Get tags if available
            print(f"✓ Cold Start: Fetched enhanced data for '{source_game_title}'.")
        else:
            # Fall back to basic API call if enhanced failed
            print(f"ℹ Cold Start: Enhanced data fetch failed or incomplete for {app_id_int}. Trying basic API call...")
            app_text, api_source_title = get_steam_app_details(app_id_int) # Basic fetcher
            source_game_title = api_source_title # Use title from basic API
            # No reliable tags from basic API call
            if app_text:
                 print(f"✓ Cold Start: Fetched basic data for '{source_game_title}'.")

        if app_text:
            try:
                # Transform the new game's text using the loaded TF-IDF model
                print(f"✓ Cold Start: Transforming text for {app_id_int} ('{source_game_title}')")
                app_vector = tfidf.transform([app_text])
                # Compute similarity against the entire existing matrix
                print(f"✓ Cold Start: Calculating similarity for {app_id_int}")
                cosine_sim = linear_kernel(app_vector, tfidf_matrix).flatten()
                print(f"✓ Cold Start: Successfully generated TF-IDF recommendations for {app_id_int}.")
            except Exception as e_process:
                print(f"✗ Cold Start: Error processing TF-IDF vector for {app_id_int}: {e_process}")
                # Try tag-based as last resort if we have tags from enhanced data
                if source_tags and len(source_tags) > 0:
                    print(f"ℹ Cold Start: Attempting tag-based fallback for {app_id_int}...")
                    tag_recs = get_tag_based_recommendations(app_id_int, df_meta, df_games, source_tags, n)
                    if not tag_recs.empty:
                        print(f"✓ Cold Start: Found {len(tag_recs)} recommendations via tag-based fallback.")
                        tag_recs['recommendation_type'] = 'content_tags_cold_start'
                        return tag_recs, source_game_title
                    else:
                         print(f"✗ Cold Start: Tag-based fallback failed for {app_id_int}.")
                return pd.DataFrame(), source_game_title # Return empty if TF-IDF and tags failed
        else:
            print(f"✗ Cold Start: Could not fetch any data via Steam API for app_id {app_id_int}.")
            return pd.DataFrame(), source_game_title # Return empty if API fails completely
        # --- End Cold Start Logic ---

    # If cosine_sim was not calculated (e.g., error before calculation)
    if cosine_sim is None:
        print(f"✗ Cosine similarity calculation failed unexpectedly for {app_id}")
        return pd.DataFrame(), source_game_title

    # Get similarity scores (index, score)
    sim_scores = list(enumerate(cosine_sim))
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Print the highest similarity score for debugging
    if sim_scores:
        highest_similarity = sim_scores[0][1]
        print(f"✓ Highest TF-IDF similarity score for {app_id_int} ('{source_game_title}'): {highest_similarity:.4f}")
    else:
        print(f"✗ No similarity scores generated for {app_id_int}.")
        return pd.DataFrame(), source_game_title

    # Exclude self if the game was found in the original dataset ('idx' is defined and game wasn't enhanced)
    # If enhanced_data was used, idx might not be relevant to the cosine_sim vector source, so don't exclude based on it.
    if 'idx' in locals() and idx is not None and not enhanced_data:
        original_length = len(sim_scores)
        sim_scores = [score for score in sim_scores if score[0] != idx]
        if len(sim_scores) < original_length:
             print(f"✓ Removed self-recommendation (index {idx})")

    # Filter by minimum similarity threshold
    original_length = len(sim_scores)
    relevant_scores = [score for score in sim_scores if score[1] >= min_similarity]
    print(f"✓ Found {len(relevant_scores)} games with similarity >= {min_similarity} (out of {original_length} potential).")

    # Decide whether to use TF-IDF or attempt tag-based fallback
    final_sim_scores = []
    recommendation_source = "content_tfidf" # Default source

    # Use TF-IDF if enough relevant scores OR if tag fallback is not possible/fails
    if len(relevant_scores) >= 3:
        print(f"✓ Using top {n} TF-IDF recommendations.")
        final_sim_scores = relevant_scores[:n]
    else:
        print(f"ℹ WARNING: Only {len(relevant_scores)} TF-IDF games found with similarity >= {min_similarity}. Trying tag-based fallback...")

        # Determine tags for fallback
        fallback_tags = None
        if enhanced_data and enhanced_data.get('tags_str'):
            fallback_tags = set(enhanced_data['tags_str'].lower().split())
            print("✓ Using tags from enhanced API data for fallback.")
        elif 'idx' in locals() and idx is not None and 'tags_str' in df_meta.columns:
            meta_tags = df_meta.loc[idx, 'tags_str']
            if pd.notna(meta_tags) and meta_tags.strip():
                fallback_tags = set(meta_tags.lower().split())
                print("✓ Using tags from local metadata for fallback.")

        # Attempt tag-based fallback if tags are available
        if fallback_tags and len(fallback_tags) > 0:
            tag_recs_df = get_tag_based_recommendations(app_id_int, df_meta, df_games, fallback_tags, n)

            if not tag_recs_df.empty and len(tag_recs_df) >= 3:
                print(f"✓ Using {len(tag_recs_df)} tag-based recommendations as fallback.")
                tag_recs_df['recommendation_type'] = 'content_tags_fallback'
                return tag_recs_df, source_game_title # Return tag results directly
            else:
                print(f"✗ Tag-based fallback failed or yielded too few results ({len(tag_recs_df)}).")
        else:
             print(f"✗ No usable tags found for tag-based fallback.")

        # If tag-based fallback failed or wasn't attempted, use the best TF-IDF scores we have, even if few
        print(f"✓ Reverting to using the top {n} available TF-IDF scores (even if below threshold or count).")
        final_sim_scores = sim_scores[:n] # Use the original sorted list before thresholding

    # --- Process final TF-IDF recommendations ---
    if not final_sim_scores:
         print(f"✗ No recommendations generated for {app_id_int} ('{source_game_title}') after all steps.")
         return pd.DataFrame(), source_game_title

    # Get recommended game indices from final scores
    game_indices = [i[0] for i in final_sim_scores]

    # Ensure indices are within bounds of df_meta
    valid_indices = [i for i in game_indices if i < len(df_meta)]
    if len(valid_indices) != len(game_indices):
        print(f"✗ Warning: {len(game_indices) - len(valid_indices)} recommendation indices were out of bounds for df_meta (length {len(df_meta)})")
        game_indices = valid_indices
        # Update final_sim_scores to match valid indices
        final_sim_scores = [score for score in final_sim_scores if score[0] in valid_indices]


    if not game_indices:
        print(f"✗ No valid recommendation indices found for {app_id} after bounds check.")
        return pd.DataFrame(), source_game_title

    # Get details for recommended games
    recommended_app_ids = df_meta.iloc[game_indices]['app_id'].values
    recommended_games = df_games[df_games['app_id'].isin(recommended_app_ids)].copy()

    # Add similarity scores from the final list
    sim_dict_by_index = {i: score for i, score in final_sim_scores}
    # Map df_meta index to app_id for the recommended games
    index_to_appid = df_meta.iloc[game_indices][['app_id']].reset_index().set_index('index')['app_id'].to_dict() # Map index -> app_id
    # Create app_id -> similarity mapping
    sim_dict_by_appid = {app_id: sim_dict_by_index[index] for index, app_id in index_to_appid.items() if index in sim_dict_by_index}

    recommended_games['similarity'] = recommended_games['app_id'].map(sim_dict_by_appid)
    recommended_games['similarity'] = recommended_games['similarity'].fillna(0) # Fill NaNs just in case
    recommended_games = recommended_games.sort_values('similarity', ascending=False)

    # Add recommendation type
    recommended_games['recommendation_type'] = recommendation_source # 'content_tfidf'

    # Ensure essential columns exist
    final_cols_ordered = ['app_id', 'title', 'similarity', 'date_release', 'rating', 'price_final', 'recommendation_type']
    for col in final_cols_ordered:
        if col not in recommended_games.columns:
            if col == 'similarity': recommended_games[col] = 0
            elif col == 'recommendation_type': recommended_games[col] = recommendation_source
            else: recommended_games[col] = None # Add missing columns like rating, price etc. if needed

    recommended_games = recommended_games[final_cols_ordered] # Reorder/select final columns

    print(f"✓ Returning {len(recommended_games)} {recommendation_source} recommendations for {app_id} ('{source_game_title}')")
    return recommended_games, source_game_title



def get_collaborative_recommendations(app_id=None, favorite_games=None, n=12, df_games=None):
    """
    Generate recommendations using the precomputed item similarity matrix.
    
    Can work in two modes:
    1. Single game mode: Recommend games similar to a specific app_id
    2. Profile mode: Recommend games based on a list of favorite_games
    
    Args:
        app_id (int, optional): A single game ID to get similar games for
        favorite_games (list, optional): A list of favorite game IDs
        n (int): Number of recommendations to return
        df_games (DataFrame, optional): DataFrame with game details, if already loaded
    
    Returns:
        DataFrame: Recommendations with app_id, title, similarity score, etc.
                   Includes a 'recommendation_type' column set to 'collaborative'.
        str: Title of the source game (if single game mode) or generic title.
    """
    global _item_similarity_cache, _df_games_cache
    
    # Validate inputs
    if app_id is None and (not favorite_games or len(favorite_games) == 0):
        print("Error: Either app_id or favorite_games must be provided")
        return pd.DataFrame(), "N/A"
    
    # Load item similarity matrix if not already in cache
    if _item_similarity_cache is None:
        try:
            print(f"Loading item similarity matrix from {ITEM_SIMILARITY_PATH}")
            _item_similarity_cache = joblib.load(ITEM_SIMILARITY_PATH)
            print(f"Loaded similarity matrix with shape {_item_similarity_cache.shape}")
        except FileNotFoundError:
            print(f"Error: Item similarity matrix not found at {ITEM_SIMILARITY_PATH}")
            return pd.DataFrame(), "N/A"
        except Exception as e:
            print(f"Error loading item similarity matrix: {e}")
            return pd.DataFrame(), "N/A"
    
    # Use the provided df_games or load from cache/function
    if df_games is None:
        if _df_games_cache is not None:
            df_games = _df_games_cache
        else:
            # Load from your existing function
            _, _, df_games_loaded, _ = load_model_and_data()
            if df_games_loaded is None:
                 print("Failed to load game data for collaborative filtering.")
                 return pd.DataFrame(), "N/A"
            _df_games_cache = df_games_loaded
            df_games = df_games_loaded # Use the loaded data
    
    if df_games is None or df_games.empty:
        print("No game data available for collaborative filtering")
        return pd.DataFrame(), "N/A"
    
    sim_matrix = _item_similarity_cache
    source_game_title = "Your Profile" # Default for favorite_games mode
    
    # Track which recommendations came from which source game
    all_recommendations = {}  # app_id -> (max_similarity, source_app_id)
    
    if app_id is not None:
        # Mode 1: Recommendations based on a single game
        try:
            app_id_int = int(app_id)
        except (ValueError, TypeError):
            print(f"Invalid app_id format: {app_id}")
            return pd.DataFrame(), f"Invalid Game ID {app_id}"

        if app_id_int not in sim_matrix.index:
            print(f"Game {app_id_int} not found in similarity matrix")
            # Try to get title from df_games as fallback
            source_game_info = df_games[df_games['app_id'] == app_id_int]
            if not source_game_info.empty:
                source_game_title = source_game_info['title'].iloc[0]
            else:
                source_game_title = f"Game ID {app_id_int}"
            return pd.DataFrame(), source_game_title
        
        # Get source game title
        source_game_info = df_games[df_games['app_id'] == app_id_int]
        if not source_game_info.empty:
            source_game_title = source_game_info['title'].iloc[0]
        else:
            source_game_title = f"Game ID {app_id_int}" # Fallback if not in df_games

        # Get similar items for this game
        similar_items = sim_matrix.loc[app_id_int]
        # Remove self-similarity
        similar_items = similar_items[similar_items.index != app_id_int]
        
        # Convert to dictionary for consistent processing
        for item_id, score in similar_items.items():
            # Don't include negative similarities or very low scores
            if score > 0.01: # Threshold can be adjusted
                all_recommendations[item_id] = (score, app_id_int)
    
    else:
        # Mode 2: Recommendations based on multiple favorite games
        valid_favorites = []
        for fav_id in favorite_games:
            try:
                fav_id_int = int(fav_id)
                if fav_id_int in sim_matrix.index:
                    valid_favorites.append(fav_id_int)
                else:
                    print(f"Favorite game {fav_id_int} not in similarity matrix")
            except (ValueError, TypeError):
                print(f"Invalid favorite game ID: {fav_id}")
        
        if not valid_favorites:
            print("No valid favorite games found in similarity matrix")
            return pd.DataFrame(), source_game_title # Return default title "Your Profile"
        
        # For each favorite game, get similar items
        for fav_id in valid_favorites:
            similar_items = sim_matrix.loc[fav_id]
            # Remove the favorite itself and other favorites from recommendations
            similar_items = similar_items[(similar_items.index != fav_id) & 
                                          ~similar_items.index.isin(valid_favorites)]
            
            # Add to recommendations dict, keeping highest score if already exists
            for item_id, score in similar_items.items():
                if score > 0.01:  # Only positive similarities above threshold
                    if item_id not in all_recommendations or score > all_recommendations[item_id][0]:
                        all_recommendations[item_id] = (score, fav_id)
    
    if not all_recommendations:
        print("No recommendations found")
        return pd.DataFrame(), source_game_title
    
    # Convert recommendations to DataFrame
    rec_data = [(rec_app_id, sim, src) for rec_app_id, (sim, src) in all_recommendations.items()]
    rec_df = pd.DataFrame(rec_data, columns=['app_id', 'similarity', 'source_game_id']) # Renamed source_game to source_game_id
    
    # Sort by similarity score
    rec_df = rec_df.sort_values('similarity', ascending=False)
    
    # Get top N recommendations
    rec_df = rec_df.head(n)
    
    # Merge with game details
    # Ensure df_games has the necessary columns
    required_cols = ['app_id', 'title', 'date_release', 'rating', 'price_final']
    if not all(col in df_games.columns for col in required_cols):
        print(f"Warning: df_games is missing one or more required columns: {required_cols}")
        # Select only available columns from df_games to merge
        available_cols = ['app_id'] + [col for col in required_cols if col in df_games.columns and col != 'app_id']
        df_games_subset = df_games[available_cols].copy() # Use copy to avoid SettingWithCopyWarning
    else:
        df_games_subset = df_games[required_cols].copy() # Use copy to avoid SettingWithCopyWarning

    # Ensure app_id types match for merging
    rec_df['app_id'] = rec_df['app_id'].astype(int)
    df_games_subset['app_id'] = df_games_subset['app_id'].astype(int)

    result = pd.merge(rec_df, df_games_subset, on='app_id', how='inner')
    
    # If no matches were found after merging
    if result.empty:
        print("No recommendations were found in the game details database")
        return pd.DataFrame(), source_game_title
        
    # Add recommendation type
    result['recommendation_type'] = 'collaborative'

    # Ensure final columns exist, adding 'similarity' from rec_df
    final_cols = required_cols + ['similarity', 'source_game_id', 'recommendation_type']
    for col in final_cols:
        if col not in result.columns:
            # Handle potential missing columns from the merge or original df_games
            if col in rec_df.columns:
                 result[col] = rec_df[col] # Get from rec_df if available (like similarity)
            elif col == 'recommendation_type':
                 result[col] = 'collaborative' # Should already be set, but as fallback
            else:
                 result[col] = None if col != 'similarity' else 0 # Default value

    # Reorder columns for clarity
    final_cols_ordered = [col for col in ['app_id', 'title', 'similarity', 'date_release', 'rating', 'price_final', 'recommendation_type', 'source_game_id'] if col in result.columns]
    result = result[final_cols_ordered]

    print(f"Generated {len(result)} collaborative filtering recommendations")
    return result, source_game_title


def get_hybrid_recommendations(app_id, n=12, df_games=None, content_weight=0.5, collab_weight=0.5):
    """
    Generates hybrid recommendations leveraging off the combination of content-based and item-item collaborative filtering.
    Uses interleaved approach to combine the two recommendation types.
    """
    print(f"Generating hybrid recommendations for app_id {app_id}...")

    # Get content-based recommendations
    content_recs_df, source_game_title_content = get_content_recommendations(app_id, n=n)
    if content_recs_df.empty:
        print(f"✗ No content-based recommendations found for {app_id}.")
        return pd.DataFrame(), source_game_title_content
    else:
        content_recs_df = content_recs_df.copy()  # Ensure we are working with a copy
        content_recs_df['source'] = 'content'
        print(f"✓ Found {len(content_recs_df)} content-based recommendations for {app_id}.")

    # Get Item-item collaborative recommendations
    # Ensure all df_games is available
    global _df_games_cache
    if df_games is None:
        if _df_games_cache is not None:
            df_games = _df_games_cache
        else:
            _, _, df_games_loaded, _ = load_model_and_data()
            if df_games_loaded is not None:
                _df_games_cache = df_games_loaded
                df_games = df_games_loaded
            else:
                print("Hybrid: Failed to load df_games for collaborative part.")
                df_games = pd.DataFrame() # Ensure it's a DataFrame

    collab_recs_df, source_game_title_collab = get_collaborative_recommendations(app_id=app_id, n=n, df_games=df_games)
    if collab_recs_df.empty:
        print(f"✗ No collaborative recommendations found for {app_id}.")
        return pd.DataFrame(), source_game_title_collab
    else:
        collab_recs_df = collab_recs_df.copy()  # Ensure we are working with a copy
        collab_recs_df['source'] = 'collaborative'
        print(f"✓ Found {len(collab_recs_df)} collaborative recommendations for {app_id}.")

    # Determine primary source title
    source_game_title = source_game_title_content if source_game_title_content and not source_game_title_content.startswith("Game ID") else source_game_title_collab

    # --- Interleaving Strategy ---
    hybrid_recs = []
    content_list = content_recs_df.to_dict('records') if not content_recs_df.empty else []
    collab_list = collab_recs_df.to_dict('records') if not collab_recs_df.empty else []
    
    added_app_ids = set()
    len_max = max(len(content_list), len(collab_list))
    
    for i in range(len_max):
        # Add content rec if available and not already added
        if i < len(content_list):
            rec = content_list[i]
            app_id_rec = rec.get('app_id')
            if app_id_rec not in added_app_ids:
                rec['hybrid_rank'] = len(hybrid_recs) + 1 # Add rank based on interleaving order
                hybrid_recs.append(rec)
                added_app_ids.add(app_id_rec)
                if len(hybrid_recs) == n: break # Stop if we reached desired count
        
        if len(hybrid_recs) == n: break # Check again after potentially adding content rec
        
        # Add collab rec if available and not already added
        if i < len(collab_list):
            rec = collab_list[i]
            app_id_rec = rec.get('app_id')
            if app_id_rec not in added_app_ids:
                rec['hybrid_rank'] = len(hybrid_recs) + 1 # Add rank
                hybrid_recs.append(rec)
                added_app_ids.add(app_id_rec)
                if len(hybrid_recs) == n: break # Stop if we reached desired count
    
    if not hybrid_recs:
        print("Hybrid: No recommendations generated after combining.")
        return pd.DataFrame(), source_game_title
        
    hybrid_df = pd.DataFrame(hybrid_recs)
    hybrid_df['recommendation_type'] = 'hybrid' # Set final type
    
    # Ensure standard columns are present
    final_cols_ordered = ['app_id', 'title', 'similarity', 'source', 'hybrid_rank', 'date_release', 'rating', 'price_final', 'recommendation_type', 'source_game_id']
    for col in final_cols_ordered:
         if col not in hybrid_df.columns:
             if col == 'similarity': hybrid_df[col] = 0.0 # Default similarity if missing
             elif col == 'hybrid_rank': hybrid_df[col] = 0 # Default rank
             else: hybrid_df[col] = None
    
    # Select and reorder columns
    present_cols = [col for col in final_cols_ordered if col in hybrid_df.columns]
    hybrid_df = hybrid_df[present_cols]
    
    print(f"--- Generated {len(hybrid_df)} HYBRID recommendations for {app_id} ('{source_game_title}') ---")
    return hybrid_df, source_game_title
