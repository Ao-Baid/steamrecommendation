from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
# Import the specific function from your item_similarity module
from steamrecommendations.item_similarity import calculate_and_save_item_similarity

class Command(BaseCommand):
    help = 'Calculates and saves the item-item similarity matrix based on positive recommendations.'

    def add_arguments(self, parser):
        # Argument for the input recommendations CSV file path
        parser.add_argument(
            '--recs-path',
            type=str,
            default=None, # Default will be handled using settings.BASE_DIR
            help='Path to the recommendations CSV file. Defaults to data/recommendations.csv within BASE_DIR.'
        )
        # Argument for the output similarity matrix file path
        parser.add_argument(
            '--output-path',
            type=str,
            default=None, # Default will be handled using settings.BASE_DIR
            help='Path to save the computed similarity matrix. Defaults to model_cache/item_similarity.joblib within BASE_DIR.'
        )
        # Argument for the app ID map path (optional, based on function signature)
        parser.add_argument(
            '--app-id-map-path',
            type=str,
            default=None, # Default will be handled using settings.BASE_DIR
            help='Path to save the app ID mapping (optional). Defaults to model_cache/item_similarity_app_id_map.joblib within BASE_DIR.'
        )

    def handle(self, *args, **options):
        # Determine default paths using Django settings
        BASE_DIR = settings.BASE_DIR
        default_recs_path = os.path.join(BASE_DIR, 'data', 'recommendations_reduced.csv')
        default_output_path = os.path.join(BASE_DIR, 'model_cache', 'item_similarity.joblib')
        default_app_id_map_path = os.path.join(BASE_DIR, 'model_cache', 'item_similarity_app_id_map.joblib')

        # Use provided paths or the defaults
        recs_path = options['recs_path'] or default_recs_path
        output_path = options['output_path'] or default_output_path
        app_id_map_path = options['app_id_map_path'] or default_app_id_map_path

        self.stdout.write(self.style.NOTICE(f"Starting item similarity calculation..."))
        self.stdout.write(f"Input recommendations file: {recs_path}")
        self.stdout.write(f"Output similarity matrix: {output_path}")
        # Only mention map path if it's actually used by the function being called
        # self.stdout.write(f"Output App ID map: {app_id_map_path}")

        try:
            # Call the imported function
            # Note: The function in the snippet doesn't return anything, so we don't check success
            # It also doesn't accept app_id_map_path in the provided snippet, adjust if your actual function does
            calculate_and_save_item_similarity(
                recs_path=recs_path,
                output_path=output_path
                # app_id_map_path=app_id_map_path # Add this if your function takes it
            )
            self.stdout.write(self.style.SUCCESS('Item similarity calculation completed successfully.'))
        except FileNotFoundError:
            raise CommandError(f"Input file not found at {recs_path}. Please check the path.")
        except MemoryError:
            raise CommandError("MemoryError: The dataset is likely too large to process with pivot_table. Consider using the sparse matrix implementation.")
        except Exception as e:
            # Catch other potential errors from the function (e.g., pandas errors)
            raise CommandError(f"An error occurred during item similarity calculation: {e}")
