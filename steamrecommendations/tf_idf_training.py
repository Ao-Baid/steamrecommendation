import pandas as pd
import os
import joblib
import re # Import regex
import nltk # Import NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from bs4 import BeautifulSoup # Import BeautifulSoup for HTML parsing


DATA_DIR = './data'
MODEL_DIR = './model_cache' # Directory to save/load the model
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_model_full.joblib') # Changed filename
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix_full.joblib') # Changed filename
GAMES_DF_PATH = os.path.join(MODEL_DIR, 'games_df_full.joblib') # Changed filename
META_DF_PATH = os.path.join(MODEL_DIR, 'meta_df_full.joblib') # Changed filename and variable
ITEM_SIMILARITY_PATH = os.path.join(MODEL_DIR, 'item_similarity.joblib') # Path for item similarity matrix
APP_ID_MAP_PATH = os.path.join(MODEL_DIR, 'item_similarity_app_id_map.joblib') # Path for app ID mapping

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

#print the amount of empty values in games_metadata.json, specifically the description column
def check_empty_values_in_metadata():
    """Checks for null, empty string, or whitespace-only descriptions in metadata."""
    try:
        df_games_meta = pd.read_json(os.path.join(DATA_DIR, 'games_metadata.json'), lines=True, orient="records")

        # 1. Count null/NaN descriptions
        null_descriptions = df_games_meta['description'].isnull().sum()

        # 2. Count descriptions that are empty strings ""
        # Fill NaN with a unique placeholder to avoid errors with string methods, then check for ""
        empty_string_descriptions = df_games_meta['description'].fillna('__NAN__').astype(str).eq('').sum()

        # 3. Count descriptions that contain only whitespace
        # Fill NaN, convert to string, strip whitespace, then check if empty
        whitespace_only_descriptions = df_games_meta['description'].fillna('__NAN__').astype(str).str.strip().eq('').sum()

        # Note: Condition 3 (whitespace-only) will also include Condition 2 (empty string)
        # If you want truly distinct counts, you might need more complex logic.
        # However, often the goal is to find *effectively* empty descriptions.
        # Let's count descriptions that are null OR effectively empty (empty string or just whitespace)

        effectively_empty_count = df_games_meta['description'].isnull() | \
                                  df_games_meta['description'].astype(str).str.strip().eq('')
        total_effectively_empty = effectively_empty_count.sum()


        print(f"Number of null/NaN descriptions: {null_descriptions}")
        # print(f"Number of empty string \"\" descriptions: {empty_string_descriptions}") # Included in whitespace count
        print(f"Number of whitespace-only or empty string descriptions: {whitespace_only_descriptions}")
        print(f"Total descriptions that are null, empty string, or whitespace-only: {total_effectively_empty}")

    except FileNotFoundError:
        print("Error: Metadata file not found.")
    except KeyError:
        print("Error: 'description' column not found in metadata file.")
    except Exception as e:
        print(f"Error checking empty values: {e}")

# Call the function (if you want it to run when the script loads)
check_empty_values_in_metadata()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

def preprocess_text(text):
    """
    Cleans HTML, tokenizes, removes stop words, and lemmatizes text.
    """
    if not isinstance(text, str):
        return ""
    # Clean HTML first
    text = _clean_html(text)
    # Remove non-alphanumeric characters and convert to lower case
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word.isalnum() and word not in stop_words and len(word) > 2 # Keep words > 2 chars
    ]
    return ' '.join(lemmatized_tokens)



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
    # Combine description and tags first
    df_processed['raw_combined_text'] = df_processed['description'].fillna('').astype(str) + ' ' + df_processed['tags_str']

    # Apply the new preprocessing function
    print("Applying NLTK preprocessing...")
    # Make sure to handle potential errors during apply if text is malformed
    try:
        df_processed['processed_text'] = df_processed['raw_combined_text'].apply(preprocess_text)
    except Exception as e:
        print(f"Error during NLTK preprocessing: {e}")
        # Optionally handle rows that failed, e.g., fill with empty string
        # df_processed['processed_text'] = df_processed['processed_text'].fillna('')
        return # Or decide how to proceed
    print("NLTK preprocessing complete.")


    print("Training TF-IDF model on full dataset...")
    # Use the preprocessed text.
    # Set lowercase=False and stop_words=None as these are handled by preprocess_text.
    tfidf = TfidfVectorizer(max_features=10000, lowercase=False, stop_words=None)
    tfidf_matrix = tfidf.fit_transform(df_processed['processed_text']) # Use processed_text

    print("Saving full model and data...")
    joblib.dump(tfidf, TFIDF_MODEL_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    # Save the full df_games and the processed metadata df (only necessary columns)
    joblib.dump(df_games, GAMES_DF_PATH)
    # Save relevant columns from df_processed, including app_id and the processed text
    # Keep description and tags_str if needed for display or other purposes
    joblib.dump(df_processed[['app_id', 'description', 'tags_str', 'processed_text']], META_DF_PATH)

    print("Full model training and saving complete.")

# Add a main execution block to run the training
if __name__ == "__main__":
    print("Starting TF-IDF model training process...")
    # Set force_retrain=True if you always want to retrain when running the script directly
    train_and_save_model(force_retrain=False)
    print("TF-IDF model training process finished.")