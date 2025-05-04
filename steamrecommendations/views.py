from django.shortcuts import render, redirect
from .forms import UserSurveyForm
from .models import SurveyUserProfile
from django.http import HttpResponse, Http404
from django.core.cache import cache
from django.urls import reverse
from django.contrib import messages
from .recommendation import get_content_recommendations, load_model_and_data, get_collaborative_recommendations
import pandas as pd
import random
import requests
import numpy as np

# Pre-load df_games if possible to speed up collaborative filtering calls
# This is a simple approach; consider more robust caching/loading strategies
DF_GAMES_CACHE = None
def preload_df_games():
    global DF_GAMES_CACHE
    if DF_GAMES_CACHE is None:
        try:
            _, _, games_df, _ = load_model_and_data()
            if games_df is not None:
                DF_GAMES_CACHE = games_df
                print("Preloaded df_games into view cache.")
            else:
                print("Failed to preload df_games.")
        except Exception as e:
            print(f"Error preloading df_games: {e}")

preload_df_games() # Load when the server starts

# Create your views here.
def index(request):
    # Check if data is already cached
    cached_games = cache.get("top100in2weeks")
    if cached_games is not None:
        games = cached_games
    else:
        # Fetch data from Steam Spy API
        response = requests.get("https://steamspy.com/api.php?request=top100in2weeks")
        games = response.json()
        if not isinstance(games, dict) or not games:
            return HttpResponse("Invalid or empty data received from Steam Spy API.", status=500)
        # Cache data for 1 hour
        cache.set('top100in2weeks', games, timeout=60*60)
    
    # Format game prices and other data
    games_list = list(games.values())
    for game in games_list:
        if float(game.get('price', 0)) == 0.0:
            game['display_price'] = "Free to Play"
            game['is_free'] = True
        else:
            game['display_price'] = f"${float(game['price']) / 100:.2f}"
            game['is_free'] = False
    
    # Select a larger pool of games for shuffling (30 games)
    display_games = random.sample(games_list, min(30, len(games_list)))
    
    return render(request, "steamrecommendations/index.html", {"games": display_games})

def about(request):
    return render(request, "steamrecommendations/about.html")

def games_list(request):
    # Check if data is already cached
    cached_games = cache.get("top100in2weeks")
    if cached_games is not None:
        games = cached_games
    else:
        # Fetch data from Steam Spy API
        response = requests.get("https://steamspy.com/api.php?request=top100in2weeks")
        games = response.json()
        # Cache data for 1 hour
        cache.set('top100in2weeks', games, timeout=60*60)
    
    # Get all games sorted by player count
    sorted_games = sorted(
        list(games.values()), 
        key=lambda x: x.get('ccu', 0), 
        reverse=True
    )
    
    # Adjust game prices
    for game in sorted_games:
        if float(game['price']) == 0.0:
            game['price'] = "Free to Play"
            game['is_free'] = True
        else:
            game['price'] = f"{float(game['price']) / 100:.2f}"
            game['is_free'] = False

    return render(request, "steamrecommendations/games_list.html", {"games": sorted_games})


def search_games(request):
    query = request.GET.get('q', '')
    results = []
    if query:
        # load the games dataframe used by the recommender
        _, _, df_games, _ = load_model_and_data()
        if df_games is not None:
            # Case-insensitive search
            search_results_df = df_games[df_games['title'].str.contains(query, case=False, na=False)]
            # Select relevant columns and limit results
            results = search_results_df[['app_id', 'title', 'date_release', 'rating', 'price_final']].head(50).to_dict(orient='records')
        else:
            messages.error(request, "Error loading game data. Please try again later.")

    return render(request, "steamrecommendations/search_results.html", {"query": query, "results": results})


def recommendations_for_game(request, app_id):
    try:
        # Ensure app_id is an integer
        app_id_int = int(app_id)
    except ValueError:
        raise Http404("Invalid game ID format.")

    recommendations_df, source_game_title = get_content_recommendations(app_id_int)

    if source_game_title is None:
         raise Http404(f"Game with ID {app_id_int} not found in recommendation dataset.")

    context = {
        'source_game_title': source_game_title,
        'source_app_id': app_id_int,
        'recommendations': recommendations_df.to_dict(orient='records') if not recommendations_df.empty else [],
    }
    return render(request, 'steamrecommendations/recommendations_for_game.html', context)


def user_survey(request):
    if request.method == 'POST':
        form = UserSurveyForm(request.POST)
        if form.is_valid():
            # Process the form data without associating it with a user
            favorite_genres = form.cleaned_data['favorite_genres']
            preferred_price_range = form.cleaned_data['preferred_price_range']
            play_time_preference = form.cleaned_data['play_time_preference']
            # You can handle the data here, e.g., save it to a temporary storage or use it directly
            return redirect('personalized_recommendations')
    else:
        form = UserSurveyForm()
    
    return render(request, 'steamrecommendations/user_survey.html', {'form': form})


def collaborative_recommendations_for_game(request, app_id):
    """View for showing collaborative filtering recommendations for a specific game."""
    try:
        app_id_int = int(app_id)
    except ValueError:
        raise Http404("Invalid game ID format.")
        
    # Get recommendations using the item similarity matrix
    recommendations_df = get_collaborative_recommendations(app_id=app_id_int, n=12)
    
    if recommendations_df.empty:
        messages.warning(request, "No collaborative recommendations found for this game.")
        # Fallback to content-based if CF fails
        recommendations_df, source_game_title = get_content_recommendations(app_id_int, n=12)
        if not recommendations_df.empty:
            messages.info(request, "Showing content-based recommendations instead.")
        else:
            messages.error(request, "No recommendations available for this game.")
            return redirect('home')  # or wherever appropriate
    else:
        # Get the source game title
        source_game_title = recommendations_df['title'].iloc[0] if 'title' in recommendations_df.columns else f"Game {app_id_int}"
        
    context = {
        'source_game_title': source_game_title,
        'source_app_id': app_id_int,
        'recommendations': recommendations_df.to_dict(orient='records') if not recommendations_df.empty else [],
        'rec_type': 'collaborative'  # To indicate the type of recommendations
    }
    
    return render(request, 'steamrecommendations/collaborative_recommendations_for_game.html', context)


def recommendations_view(request, app_id):
    """
    Displays recommendations for a given app_id.
    Uses query parameter 'type' to switch between 'content' and 'collaborative'.
    Defaults to 'content'.
    """
    rec_type = request.GET.get('type', 'content').lower() # Default to content
    recommendations_df = pd.DataFrame()
    source_game_title = f"Game ID {app_id}"
    error_message = None
    recommendations_list = [] # Initialize recommendations_list before the try block

    try:
        app_id_int = int(app_id) # Validate app_id early

        if rec_type == 'collaborative':
            print(f"Getting COLLABORATIVE recommendations for {app_id_int}")
            # Pass the preloaded df_games if available
            recommendations_df, source_game_title = get_collaborative_recommendations(
                app_id=app_id_int,
                n=12,
                df_games=DF_GAMES_CACHE
            )
            if recommendations_df.empty:
                 # Try content-based as fallback if collaborative fails? Or show error.
                 error_message = f"Could not find collaborative recommendations for '{source_game_title}'. The game might not be in the interaction dataset."
                 # Optionally, fallback to content:
                 # print("Collaborative failed, falling back to content...")
                 # rec_type = 'content'

        # Use 'elif' to avoid running content if collaborative succeeded,
        # or 'if' if you want content as a fallback.
        if rec_type == 'content': # Changed to 'if' for fallback potential
             if error_message is None: # Only run if collaborative didn't already fail
                print(f"Getting CONTENT recommendations for {app_id_int}")
                recommendations_df, source_game_title = get_content_recommendations(
                    app_id=app_id_int,
                    n=12
                )
                if recommendations_df.empty and error_message is None:
                    error_message = f"Could not find content recommendations for '{source_game_title}'. The game might not be in the metadata or API fetch failed."

        # Convert DataFrame to list of dictionaries for the template
        # Handle potential NaT or NaN values for JSON serialization if needed
        # recommendations_list = [] # Moved initialization outside the try block
        if not recommendations_df.empty:
             # Replace NaN with None for template rendering
             recommendations_df = recommendations_df.replace({pd.NA: None, np.nan: None})
             recommendations_list = recommendations_df.to_dict('records')


    except ValueError:
        error_message = f"Invalid App ID provided: {app_id}"
        # raise Http404(error_message) # Changed to render with error message instead of 404
    except Exception as e:
        print(f"An unexpected error occurred in recommendations_view: {e}")
        error_message = "An error occurred while generating recommendations."
        # Optionally re-raise or handle differently depending on desired user experience
        # raise Http404(error_message) # Or render with error message

    context = {
        'source_game_title': source_game_title,
        'app_id': app_id,
        'recommendations': recommendations_list,
        'recommendation_type': rec_type, # Pass the requested type
        'error_message': error_message,
    }
    return render(request, 'steamrecommendations/recommendations_display.html', context)

# Add a view for profile-based collaborative recommendations if needed
def profile_recommendations_view(request):
     # Assuming favorite_games comes from user session, profile, or request
     favorite_games = request.session.get('favorite_games', []) # Example: Get from session
     if not favorite_games:
         # Handle case where user has no favorites
         context = {'error_message': "Add some games to your favorites to get profile recommendations!"}
         return render(request, 'steamrecommendations/recommendations_display.html', context)

     recommendations_df, source_game_title = get_collaborative_recommendations(
         favorite_games=favorite_games,
         n=12,
         df_games=DF_GAMES_CACHE
     )
     recommendations_list = []
     if not recommendations_df.empty:
         recommendations_df = recommendations_df.replace({pd.NA: None, np.nan: None})
         recommendations_list = recommendations_df.to_dict('records')

     context = {
        'source_game_title': source_game_title, # Will be "Your Profile"
        'app_id': None, # No single source app_id
        'recommendations': recommendations_list,
        'recommendation_type': 'collaborative_profile',
        'error_message': None if not recommendations_df.empty else "Could not generate profile recommendations.",
     }
     return render(request, 'steamrecommendations/recommendations_display.html', context)
