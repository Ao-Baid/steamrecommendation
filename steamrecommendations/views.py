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


def recommended_games(request):
    # Check if recommendations are already cached
    cached_recommendations = cache.get("cached_recommendations")
    if cached_recommendations is not None:
        # If cached, randomly select 10 games from the cached recommendations
        top_games = random.sample(cached_recommendations, min(10, len(cached_recommendations)))
    else:
        # Load the recommendations.csv file in chunks
        recommendations_path = "./data/recommendations.csv"
        recommendations_df = pd.read_csv(recommendations_path, usecols=['app_id', 'is_recommended'], dtype={'app_id': 'int32', 'is_recommended': 'bool'})

        # Filter out rows where 'is_recommended' is not True
        recommendations_df = recommendations_df[recommendations_df['is_recommended'] == True]

        # Count how many times each game is recommended
        recommended_counts = recommendations_df.groupby('app_id').size().reset_index(name='recommendation_count')

        # Load the games.csv file to get game details
        games_path = "./data/games.csv"
        games_df = pd.read_csv(games_path, usecols=['app_id', 'title', 'date_release', 'rating', 'price_final'], dtype={'app_id': 'int32', 'title': 'string', 'rating': 'string', 'price_final': 'float32'})

        # Merge the recommendation counts with the games data
        merged_df = pd.merge(games_df, recommended_counts, how='inner', left_on='app_id', right_on='app_id')

        # Sort games by recommendation count in descending order
        sorted_games = merged_df.sort_values(by='recommendation_count', ascending=False)

        # Convert the top games to a dictionary
        cached_recommendations = sorted_games.to_dict(orient='records')

        # Cache the recommendations for 1 hour
        cache.set("cached_recommendations", cached_recommendations, timeout=60 * 60)

        # Randomly select 10 games from the cached recommendations
        top_games = random.sample(cached_recommendations, min(10, len(cached_recommendations)))

    # Render the template with the top games
    return render(request, "steamrecommendations/recommended_games.html", {"games": top_games})

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
