from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, HTML

# Define choices for the form fields
# You can expand this list or make it dynamic if needed
COMMON_GENRES = [
    ('Action', 'Action'),
    ('Adventure', 'Adventure'),
    ('Strategy', 'Strategy'),
    ('RPG', 'RPG'),
    ('Indie', 'Indie'),
    ('Simulation', 'Simulation'),
    ('Casual', 'Casual'),
    ('Free to Play', 'Free to Play'),
    ('Sports', 'Sports'),
    ('Racing', 'Racing'),
    ('MMO', 'MMO'), # Massively Multiplayer
    # Add more genres as you see fit
]

PRICE_RANGES = [
    ('any', 'Any Price'),
    ('low', 'Low (e.g., under $10)'),
    ('medium', 'Medium (e.g., $10 - $30)'),
    ('high', 'High (e.g., over $30)'),
]

class UserSurveyForm(forms.Form):
    favorite_genres = forms.MultipleChoiceField(
        choices=COMMON_GENRES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label="Select your favorite game genres:"
    )
    preferred_price_range = forms.ChoiceField(
        choices=PRICE_RANGES,
        widget=forms.RadioSelect, # Or forms.Select for a dropdown
        required=False,
        initial='any',
        label="What's your preferred price range?"
    )
    # You could add other preferences here, e.g.:
    # play_time_preference = forms.ChoiceField(
    #     choices=[('any', 'Any'), ('short', 'Short (1-10 hours)'), ('medium', 'Medium (10-50 hours)'), ('long', 'Long (50+ hours)')],
    #     widget=forms.RadioSelect,
    #     required=False,
    #     initial='any',
    #     label="Preferred game length/play time:"
    # )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_action = 'user_survey' # Or the appropriate URL name
        self.helper.layout = Layout(
            HTML("""
                <p class="mb-3">Help us understand your gaming tastes to provide better recommendations.</p>
            """),
            Row(
                Column('favorite_genres', css_class='form-group col-md-12 mb-3'),
                css_class='mb-3'
            ),
            Row(
                Column('preferred_price_range', css_class='form-group col-md-12 mb-3'),
                css_class='mb-3'
            ),
            # Add other fields to the layout if you include them
            # Row(
            #     Column('play_time_preference', css_class='form-group col-md-12 mb-3'),
            #     css_class='mb-3'
            # ),
            Submit('submit', 'Save Preferences', css_class='btn btn-success mt-3')
        )

class GameSearchForm(forms.Form):
    query = forms.CharField(
        label='Search for a game by title', 
        max_length=100,
        widget=forms.TextInput(attrs={'placeholder': 'E.g., Cyberpunk 2077'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'get' # Typically search is GET
        self.helper.form_action = 'search_games' # URL name for search results
        self.helper.layout = Layout(
            Row(
                Column('query', css_class='form-group col-md-9 mb-0'),
                Column(Submit('submit', 'Search', css_class='btn btn-primary w-100'), css_class='form-group col-md-3 mb-0 align-self-end d-flex'),
                css_class='align-items-end' # Align items to the bottom for better visual
            )
        )