import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
from scipy.sparse import csr_matrix # Import sparse matrix type




def reduce_recommendations_dataset(recs_path='data/recommendations.csv', output_path='data/recommendations_reduced.csv', min_helpful=1):
    """
    Reduce the recommendations dataset through strategic filtering:
    - Keep only reviews with at least min_helpful votes
    - Include both positive and negative reviews
    - Keep helpful, funny, hours columns for richer signal
    - Drop unnecessary columns (date, review_id)
    """
    
    # Process in chunks to avoid loading all 41M rows at once
    chunk_size = 500000
    # Include the additional meaningful columns
    cols_to_use = ['user_id', 'app_id', 'is_recommended', 'helpful', 'funny', 'hours']
    dtype_spec = {
        'user_id': 'int64', 
        'app_id': 'int32', 
        'is_recommended': 'bool',
        'helpful': 'int32',
        'funny': 'int32',
        'hours': 'float32'
    }
    
    reader = pd.read_csv(recs_path, chunksize=chunk_size, 
                         usecols=cols_to_use,
                         dtype=dtype_spec)
    
    processed_chunks = []
    total_rows = 0
    
    for chunk in reader:
        # Keep only reviews with at least min_helpful votes - filtering for quality reviews
        chunk = chunk[chunk['helpful'] >= min_helpful]
        
        # Fill missing values in numeric columns
        chunk['funny'] = chunk['funny'].fillna(0).astype('int32')
        chunk['hours'] = chunk['hours'].fillna(0).astype('float32')
        
        # Filter users with fewer than 2 reviews (helps reduce noise)
        user_counts = chunk['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 2].index
        chunk = chunk[chunk['user_id'].isin(valid_users)]
        
        # Filter games with fewer than 5 reviews total (helps reduce very niche games)
        game_counts = chunk['app_id'].value_counts()
        popular_games = game_counts[game_counts >= 5].index
        chunk = chunk[chunk['app_id'].isin(popular_games)]
        
        processed_chunks.append(chunk)
        total_rows += len(chunk)
        print(f"Processed chunk, current total: {total_rows:,} rows")
        
        # Optional: Stop after reaching a target number of rows
        if total_rows >= 10_000_000:  # 10M rows is still a large but manageable dataset
            break
    
    # Combine and save the reduced dataset
    result = pd.concat(processed_chunks)
    result.to_csv(output_path, index=False)
    print(f"Reduced dataset saved with {len(result):,} rows")
    return output_path

reduce_recommendations_dataset(recs_path='data/recommendations.csv', output_path='data/recommendations_reduced.csv', min_helpful=1)






def calculate_and_save_item_similarity(recs_path='data/recommendations_reduced.csv',
                                       output_path='model_cache/item_similarity.joblib',
                                       app_id_map_path='model_cache/item_similarity_app_id_map.joblib'):
    """
    Calculates item-item similarity from positive recommendations using sparse matrices
    and saves the results.

    Args:
        recs_path (str): Path to the recommendations CSV file.
        output_path (str): Path to save the computed similarity matrix (DataFrame).
        app_id_map_path (str): Path to save the mapping from app_id to matrix index.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Loading recommendations from {recs_path}...")
    try:
        # Load only necessary columns and specify types for efficiency
        dtype_spec = {'user_id': 'int64', 'app_id': 'int32', 'is_recommended': 'boolean'}
        use_cols = ['user_id', 'app_id', 'is_recommended']
        df_recs = pd.read_csv(recs_path, usecols=use_cols, dtype=dtype_spec)
    except FileNotFoundError:
        print(f"Error: Recommendations file not found at {recs_path}")
        return False
    except ValueError as e:
        print(f"Error reading CSV, check columns/types: {e}")
        return False
    except Exception as e:
        print(f"Error loading recommendations CSV: {e}")
        return False

    # --- Data Preprocessing ---
    print("Preprocessing data...")
    # Drop rows with missing user_id or app_id
    df_recs.dropna(subset=['user_id', 'app_id'], inplace=True)
    # Ensure IDs are integers
    df_recs['user_id'] = df_recs['user_id'].astype(int)
    df_recs['app_id'] = df_recs['app_id'].astype(int)

    # Filter for positive recommendations only (is_recommended == True)
    # Convert boolean NA to False before filtering
    df_recs = df_recs[df_recs['is_recommended'].fillna(False)].copy()

    if df_recs.empty:
        print("No positive recommendations found after filtering. Cannot proceed.")
        return False

    # Assign a value of 1 for the interaction (since we filtered for positive recs)
    df_recs['interaction'] = 1

    # --- Create Sparse User-Item Matrix ---
    print("Creating sparse user-item matrix...")

    # Create mappings from original IDs to matrix indices (0-based)
    user_ids = df_recs['user_id'].unique()
    app_ids = df_recs['app_id'].unique()

    user_id_map = {id: i for i, id in enumerate(user_ids)}
    app_id_map = {id: i for i, id in enumerate(app_ids)}

    # Inverse map for later (matrix index back to app_id)
    app_id_map_inverse = {i: id for id, i in app_id_map.items()}

    # Handle potential duplicates (user reviewed same game twice positively)
    # Keep only one positive interaction per user-item pair
    grouped = df_recs.groupby(['user_id', 'app_id'])['interaction'].first().reset_index()

    # Map the IDs in the grouped DataFrame to the matrix indices
    user_indices = grouped['user_id'].map(user_id_map)
    item_indices = grouped['app_id'].map(app_id_map)
    interaction_values = grouped['interaction'] # Should all be 1

    try:
        # Create the sparse matrix (CSR format is good for row slicing/operations)
        user_item_sparse = csr_matrix((interaction_values, (user_indices, item_indices)),
                                      shape=(len(user_ids), len(app_ids)))
    except Exception as e:
        print(f"Error creating sparse matrix: {e}")
        return False

    # --- Item-Item Similarity Calculation ---
    # Similarity calculation needs items as rows, so transpose the user-item matrix
    item_user_sparse = user_item_sparse.T.tocsr() # Transpose and ensure CSR format
    print(f"Calculating item-item cosine similarity on sparse matrix with shape: {item_user_sparse.shape}...")

    try:
        # Calculate cosine similarity. Output is dense by default.
        # Use dense_output=False if memory is still an issue during this step,
        # but converting to DataFrame later will require dense format.
        item_similarity_matrix_dense = cosine_similarity(item_user_sparse)
    except MemoryError:
        print("MemoryError during cosine similarity calculation. The item matrix might still be too large.")
        return False
    except Exception as e:
         print(f"Error calculating cosine similarity: {e}")
         return False

    # Convert the similarity matrix to a DataFrame for easier lookup
    # Use the inverse map to get original app_ids for index/columns
    original_app_ids_ordered = [app_id_map_inverse[i] for i in range(len(app_ids))]
    df_item_similarity = pd.DataFrame(item_similarity_matrix_dense, index=original_app_ids_ordered, columns=original_app_ids_ordered)

    # --- Save the results ---
    print(f"Saving item similarity matrix to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        joblib.dump(df_item_similarity, output_path)
        # Save the app_id map (maps original app_id to matrix index 0..N-1)
        joblib.dump(app_id_map, app_id_map_path)
        print(f"App ID map saved to {app_id_map_path}")
    except Exception as e:
        print(f"Error saving similarity matrix or map: {e}")
        return False

    print("Item similarity calculation complete.")
    return True

