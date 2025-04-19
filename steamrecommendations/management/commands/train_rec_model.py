from django.core.management.base import BaseCommand, CommandError
from ...recommendation import train_and_save_model, DATA_DIR, MODEL_DIR
import os

class Command(BaseCommand):
    help = 'Train the recommendation model and save it to the model cache.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force-retrain',
            action='store_true',
            help='Force retraining of the model even if it already exists.'
        )
        parser.add_argument(
            '--sample-size',
            type=int,
            default=20000,
            help='Number of samples to use for training the model.'
        )

    def handle(self, *args, **options):
        force_retrain = options['force_retrain']
        sample_size = options['sample_size']

        self.stdout.write(self.style.NOTICE(f"Starting recommendation model training..."))
        self.stdout.write(f"Data directory: {os.path.abspath(DATA_DIR)}")
        self.stdout.write(f"Model cache directory: {os.path.abspath(MODEL_DIR)}")
        self.stdout.write(f"Sample size: {sample_size}")
        if force_retrain:
            self.stdout.write(self.style.WARNING("Forcing retraining."))

        try:
            train_and_save_model(sample_size=sample_size, force_retrain=force_retrain)
            self.stdout.write(self.style.SUCCESS('Successfully trained and saved the recommendation model.'))
        except FileNotFoundError as e:
             raise CommandError(f"Error during training: Data file not found. Ensure '{os.path.abspath(DATA_DIR)}' contains the necessary CSV/JSON files. Details: {e}")
        except Exception as e:
            raise CommandError(f"An unexpected error occurred during training: {e}")
