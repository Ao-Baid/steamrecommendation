from django import forms
from .models import SurveyUserProfile


class UserSurveyForm(forms.ModelForm):
    PRICE_CHOICES = [
        ('low', 'Under $10'),
        ('medium', '$10-30'),
        ('high', 'Over $30'),
        ('any', 'Any price range')
    ]
    
    favorite_genres = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        label="Favorite Genres"
    )

    preferred_price_range = forms.ChoiceField(
        choices=PRICE_CHOICES,
        label="Preferred Price Range"
    )

    class Meta:
        model = SurveyUserProfile
        fields = ['favourite_genres', 'preferred_price_range']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically set the choices for favorite genres based on your Genre model
        from collections import Counter
        import pandas as pd

        df_games_meta = pd.read_json('./data/games_metadata.json', lines=True, orient="records")
        genres = df_games_meta['tags'].explode().dropna().unique()
        genre_counts = Counter(genres)
        top_genres = [genre for genre, count in genre_counts.most_common(20)]

        self.fields['favorite_genres'].choices = [(genre, genre) for genre in top_genres]