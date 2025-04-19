from django.core.management.base import BaseCommand, CommandError
# Adjusted import path assuming recommendation.py is in the parent directory of management/commands/
from steamrecommendations.recommendation import train_and_save_model, DATA_DIR, MODEL_DIR
import os

class Command(BaseCommand):
    help = 'Train the recommendation model on the FULL dataset and save it to the model cache.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force-retrain',
            action='store_true',
            help='Force retraining of the model even if the full model files already exist.'
        )
        # Removed sample_size argument as the function now always uses the full dataset

    def handle(self, *args, **options):
        force_retrain = options['force_retrain']
        # Removed sample_size variable

        self.stdout.write(self.style.NOTICE(f"Starting recommendation model training on the FULL dataset..."))
        self.stdout.write(f"Data directory: {os.path.abspath(DATA_DIR)}")
        self.stdout.write(f"Model cache directory: {os.path.abspath(MODEL_DIR)}")
        # Removed sample size output
        if force_retrain:
            self.stdout.write(self.style.WARNING("Forcing retraining."))

        try:
            # Call train_and_save_model without sample_size
            train_and_save_model(force_retrain=force_retrain)
            self.stdout.write(self.style.SUCCESS('Successfully trained and saved the full recommendation model.'))
        except FileNotFoundError as e:
             # Improved error message for clarity
             raise CommandError(f"Error during training: Data file not found. Ensure '{os.path.abspath(DATA_DIR)}' contains 'games.csv' and 'games_metadata.json'. Details: {e}")
        except Exception as e:
            # Include traceback for better debugging if needed, or keep it simple
            # import traceback
            # traceback.print_exc()
            raise CommandError(f"An unexpected error occurred during training: {e}")
