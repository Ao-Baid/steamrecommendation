from django.shortcuts import render
from django.http import HttpResponse
from django.core.cache import cache
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
        #cache data for 1 hour
        cache.set('top100in2weeks', games, timeout=60)

    #select random subset of games to display
    random_games = random.sample(list(games.values()), 5)

    return render(request, "steamrecommendations/index.html", {"games": random_games})