from django import forms
from .models import SurveyUserProfile
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, Fieldset, Div


class UserSurveyForm(forms.ModelForm):
    PRICE_CHOICES = [
        ('low', 'Under $10'),
        ('medium', '$10-30'),
        ('high', 'Over $30'),
        ('any', 'Any price range')
    ]
    
    favorite_genres = forms.MultipleChoiceField(
        choices=[],
        label="Favorite Genres",
        widget=forms.SelectMultiple(attrs={'class': 'form-select', 'size': '5'}),
        help_text="Hold Ctrl/Cmd to select multiple genres"
    )

    preferred_price_range = forms.ChoiceField(
        choices=PRICE_CHOICES,
        label="Preferred Price Range",
        widget=forms.Select(attrs={'class': 'form-select'})
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
        
        # Add crispy forms helper
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-md-3'
        self.helper.field_class = 'col-md-9'
        
        self.helper.layout = Layout(
            Fieldset(
                'Gaming Preferences',
                Div(
                    'favorite_genres', 
                    css_class='mb-4'
                ),
                Div(
                    'preferred_price_range',
                    css_class='mb-4'
                ),
            ),
            Div(
                Submit('submit', 'Submit', css_class='btn btn-primary'),
                css_class='text-center mt-4'
            )
        )