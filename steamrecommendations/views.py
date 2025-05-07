from django.shortcuts import render, redirect
from .forms import UserSurveyForm
from .models import SurveyUserProfile
from django.http import HttpResponse, Http404, JsonResponse
from django.core.cache import cache
from django.urls import reverse
from django.contrib import messages
from .recommendation import get_content_recommendations, load_model_and_data, get_collaborative_recommendations, get_hybrid_recommendations, get_personalized_recommendations
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
        print("Preloading df_games for views...")
        _, _, games_df, _ = load_model_and_data() # Assuming load_model_and_data returns df_games
        if games_df is not None:
            DF_GAMES_CACHE = games_df.copy() # Store a copy
            # Ensure 'app_id' and 'title' exist for search
            if 'app_id' not in DF_GAMES_CACHE.columns or 'title' not in DF_GAMES_CACHE.columns:
                 print("Warning: DF_GAMES_CACHE is missing 'app_id' or 'title' columns.")
                 DF_GAMES_CACHE = None # Invalidate if essential columns are missing
            else:
                 print(f"df_games preloaded with {len(DF_GAMES_CACHE)} games.")
        else:
            print("Failed to preload df_games.")

preload_df_games() # Load when the server starts

def ajax_search_games(request):
    query = request.GET.get('q', '')
    results = []
    # Ensure DF_GAMES_CACHE is loaded and has necessary columns
    if query and DF_GAMES_CACHE is not None and 'app_id' in DF_GAMES_CACHE.columns and 'title' in DF_GAMES_CACHE.columns:
        # Case-insensitive search
        search_results_df = DF_GAMES_CACHE[DF_GAMES_CACHE['title'].str.contains(query, case=False, na=False)]
        # Select relevant columns and limit results
        results = search_results_df[['app_id', 'title']].head(10).to_dict(orient='records')
    return JsonResponse({'games': results})

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
            favorite_game_ids_str = request.POST.get('favorite_game_ids', '') # Expecting comma-separated IDs
            favorite_game_ids = []
            if favorite_game_ids_str:
                try:
                    # Filter out empty strings that might result from split and ensure they are digits
                    favorite_game_ids = [int(gid.strip()) for gid in favorite_game_ids_str.split(',') if gid.strip().isdigit()]
                except ValueError:
                    messages.error(request, "There was an issue with the format of favorite game IDs.")
                    # Potentially re-render form with an error or handle as appropriate

            request.session['user_preferences'] = {
                'favorite_genres': form.cleaned_data.get('favorite_genres', []),
                'preferred_price_range': form.cleaned_data.get('preferred_price_range', 'any'),
                'favorite_games': favorite_game_ids # Store list of IDs
            }
            messages.success(request, "Your preferences have been saved!")
            return redirect('personalized_recommendations_view')
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        initial_prefs = request.session.get('user_preferences', {})
        form = UserSurveyForm(initial=initial_prefs)

    existing_favorite_game_ids = initial_prefs.get('favorite_games', [])
    existing_favorite_games_details = []
    if existing_favorite_game_ids and DF_GAMES_CACHE is not None and 'app_id' in DF_GAMES_CACHE.columns and 'title' in DF_GAMES_CACHE.columns:
        try:
            # Ensure IDs are integers for isin
            valid_ids = [int(gid) for gid in existing_favorite_game_ids if str(gid).isdigit()]
            if valid_ids:
                fav_games_df = DF_GAMES_CACHE[DF_GAMES_CACHE['app_id'].isin(valid_ids)]
                existing_favorite_games_details = fav_games_df[['app_id', 'title']].to_dict('records')
        except Exception as e:
            print(f"Error fetching details for existing favorite games: {e}")


    return render(request, 'steamrecommendations/user_survey.html', {
        'form': form,
        'existing_favorite_games': existing_favorite_games_details
    })

def personalized_recommendations_view(request):
    user_preferences = request.session.get('user_preferences')
    if not user_preferences:
        messages.warning(request, "Please complete the survey to get personalized recommendations.")
        return redirect('user_survey')

    # Ensure favorite_games is part of user_preferences if not already
    if 'favorite_games' not in user_preferences:
        user_preferences['favorite_games'] = [] # Default to empty list

    recommendations_df, source_game_title = get_personalized_recommendations(user_preferences, n=12)
    
    recommendations_list = []
    error_message = None

    if recommendations_df.empty:
        error_message = source_game_title 
    else:
        recommendations_df = recommendations_df.replace({pd.NA: None, np.nan: None})
        if 'date_release' in recommendations_df.columns:
            recommendations_df['date_release'] = recommendations_df['date_release'].astype(str).replace({'NaT': None, 'nan': None})
        recommendations_list = recommendations_df.to_dict('records')

    context = {
        'source_game_title': "Personalized For You", # Or use the title from the function if more specific
        'app_id': None, 
        'recommendations': recommendations_list,
        'recommendation_type': 'personalized', # Used for display logic in template
        'error_message': error_message if recommendations_df.empty else None,
        'user_preferences': user_preferences 
    }
    return render(request, 'steamrecommendations/recommendations_display.html', context)


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

            
        elif rec_type == 'hybrid':
            print(f"Getting HYBRID recommendations for {app_id_int}")
            recommendations_df, source_game_title = get_hybrid_recommendations(
                app_id=app_id_int,
                n=12,
                df_games=DF_GAMES_CACHE
            )
            if recommendations_df.empty:
                error_message = f"Could not find hybrid recommendations for '{source_game_title}'. The game might not be in the interaction dataset."


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
             # Replace NaN/NaT with None for template rendering
             recommendations_df = recommendations_df.replace({pd.NA: None, np.nan: None})
             # Convert specific columns if necessary (e.g., dates to strings)
             if 'date_release' in recommendations_df.columns:
                 # Ensure it's string or None, handle potential Timestamps
                 recommendations_df['date_release'] = recommendations_df['date_release'].astype(str).replace({'NaT': None, 'nan': None})
             recommendations_list = recommendations_df.to_dict('records')


    except ValueError:
        error_message = f"Invalid App ID provided: {app_id}"
        # raise Http404(error_message) # Changed to render with error message instead of 404
    except Exception as e:
        print(f"An unexpected error occurred in recommendations_view: {e}")
        import traceback
        traceback.print_exc() 
        error_message = "An error occurred while generating recommendations."


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
