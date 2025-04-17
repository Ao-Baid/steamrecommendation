from django.shortcuts import render, redirect
from .forms import UserSurveyForm
from .models import SurveyUserProfile
from django.http import HttpResponse
from django.core.cache import cache
import pandas as pd
import random
import requests

# Create your views here.
def index(request):
    #firstly, check if data is already cached
    cached_games = cache.get("top100in2weeks")
    if cached_games is not None:
        games = cached_games
    else:
        #fetch data from Steam Spy API
        response = requests.get("https://steamspy.com/api.php?request=top100in2weeks")
        games = response.json()
        if not isinstance(games, dict) or not games:
            return HttpResponse("Invalid or empty data received from Steam Spy API.", status=500)
        #cache data for 1 hour
        cache.set('top100in2weeks', games, timeout=60)

    #select random subset of games to display
    random_games = random.sample(list(games.values()), 5)

    return render(request, "steamrecommendations/index.html", {"games": random_games})

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
