import argparse
from persistence import PersistenceManager
from engine import Engine

def main():
    parser = argparse.ArgumentParser(description='Autopoietic Probability Simulation')

    parser.add_argument('--consensus-strategy', type=str, default='superposition',
                        choices=['product', 'superposition'],
                        help='Consensus formation strategy (default: superposition)')

    parser.add_argument('--dimensions-concurrency', type=int, default=None,
                        help='Number of semantic dimensions to process concurrently (default: all)')

    parser.add_argument('--dimensions-strategy', type=str, default='independent',
                        choices=['independent', 'coupled'],
                        help='Strategy for dimension processing (default: independent)')

    parser.add_argument('--persistence-enabled', action='store_true', default=False,
                        help='Enable state persistence (default: False)')

    parser.add_argument('--csv-export-enabled', action='store_true', default=False,
                        help='Enable CSV data export (default: False)')

    parser.add_argument('--perspectives-count', type=int, default=None,
                        help='Number of perspectives to include (default: all available)')

    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducible results (default: system time)')

    parser.add_argument('--iterations-count-max', type=int, default=100_000,
                        help='The number of iterations to run (default: 100,000)')

    args = parser.parse_args()

    # Prepare engine parameters
    engine_kwargs = {
        'consensus_strategy': args.consensus_strategy,
        'dimensions_strategy': args.dimensions_strategy,
        'csv_export_enabled': args.csv_export_enabled,
        'random_seed': args.random_seed,
        'iterations_count_max': args.iterations_count_max
    }

    # Only set dimensions_concurrency if specified
    if args.dimensions_concurrency is not None:
        engine_kwargs['dimensions_concurrency'] = args.dimensions_concurrency

    # Only set perspectives_count if specified
    if args.perspectives_count is not None:
        engine_kwargs['perspectives_count'] = args.perspectives_count

    persistence_manager = None
    if args.persistence_enabled:
        # Create persistence manager
        persistence_manager = PersistenceManager("state.pkl", fsync_interval=10000)

    # Try to load from persistence first
    engine = Engine.load_from_persistence(persistence_manager, **engine_kwargs)

    engine.start()

if __name__ == "__main__":
    main()
