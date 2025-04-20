from django.db import models
from django.contrib.auth.models import User

class SurveyUserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    favourite_genres = models.JSONField(default=list)  # Store favorite genres as a list of strings
    preferred_price_range = models.CharField(max_length=50, blank=True)  # e.g., "0-20", "20-50"
    preferred_moods = models.JSONField(default=list)  # Store preferred moods as a list of strings
    preferred_platforms = models.JSONField(default=list)  # Store preferred platforms as a list of strings
    preferred_themes = models.JSONField(default=list)  # Store preferred themes as a list of strings
    favourite_games = models.JSONField(default=list)  # Store favorite games as a list of game IDs
    play_time_preference = models.CharField(max_length=50, blank=True)  # e.g., "casual", "hardcore"

    def __str__(self):
        return f"{self.user.username}'s Profile"
    

