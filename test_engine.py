#!/usr/bin/env python3

import os
import unittest
from decimal import Decimal
from unittest.mock import MagicMock
from engine import Engine, SEMANTICS, Perspectives, TheSignal, PersistenceManager, Iteration, Consensus
from csv_export import CSVExporter

class TestEngine(unittest.TestCase):
    def test_defaults(self):
        engine = Engine()

        self.assertEqual(engine.consensus_strategy, "superposition")
        self.assertEqual(engine.dimensions_concurrency, 3)
        self.assertEqual(engine.dimensions_strategy, "coupled")
        self.assertEqual(engine.csv_export_enabled, False)
        self.assertEqual(engine.perspectives_count, len(Perspectives))
        self.assertEqual(engine.random_seed, None)
        self.assertEqual(engine.iterations_count_max, 100_000)

        print("✅ engine defaults")

    def test_arguments(self):
        consensus_strategy = "product"
        dimensions_concurrency = 3
        dimensions_strategy = "coupled"
        csv_export_enabled = True
        perspectives_count = 2
        random_seed = 42
        iterations_count_max = 123

        engine = Engine(
            consensus_strategy=consensus_strategy,
            dimensions_concurrency=dimensions_concurrency,
            dimensions_strategy=dimensions_strategy,
            csv_export_enabled=csv_export_enabled,
            perspectives_count=perspectives_count,
            random_seed=random_seed,
            iterations_count_max=iterations_count_max
        )

        self.assertEqual(engine.consensus_strategy, "product")
        self.assertEqual(engine.dimensions_concurrency, 3)
        self.assertEqual(engine.dimensions_strategy, "coupled")
        self.assertEqual(engine.csv_export_enabled, True)
        self.assertEqual(engine.perspectives_count, 2)
        self.assertEqual(engine.random_seed, 42)
        self.assertEqual(engine.iterations_count_max, 123)
        self.assertEqual(len(engine.perspectives), 2)
        self.assertEqual(engine.iteration_count, 0)
        self.assertEqual(len(engine.iterations), 0)

        print("✅ engine initialization")

    def test_csv_exporter_enabled(self):
        engine = Engine(
            csv_export_enabled=True
        )

        self.assertIsInstance(engine.csv_exporter,  CSVExporter)

        print("✅ engine csv exporter instantiation")

    def test_signal(self):
        engine = Engine()

        self.assertEqual(engine.signal_perspective, TheSignal)

        print("✅ engine signal perspective")

    def test_load_from_persistence(self):
        perspectives = Perspectives[:3]
        iteration = Iteration(
            id = 111,
            responses = [],
            signal = 1.23,
            consensus = [1.11, 2.22, 3.33],
            normalized_consensus = [0.2775, 0.555, 0.8325]
        )

        iterations = []
        iteration_count = 111

        mock_persistence_manager = MagicMock(spec=PersistenceManager)
        mock_persistence_manager.load_state.return_value = [perspectives, iterations, iteration_count]

        engine = Engine.load_from_persistence(persistence_manager=mock_persistence_manager)

        self.assertEqual(engine.iterations, iterations)
        self.assertEqual(engine.iteration_count, iteration_count)
        self.assertEqual(engine.persistence_manager, mock_persistence_manager)
        self.assertEqual(engine.perspectives, perspectives)

        print("✅ engine persistence manager state load")

    def test_start(self):
        iterations_count_max = 10

        engine = Engine(iterations_count_max=iterations_count_max, random_seed=1)
        iterations = engine.start()

        self.assertEqual(len(iterations), 10)
        self.assertEqual(iterations[-1].signal, [Decimal('0.830035693274327'), Decimal('0.670305566414071'), Decimal('0.3033685109329176')])
        self.assertEqual(iterations[-1].consensus, [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, Decimal('0.4395162843688593333333333333'), Decimal('0.4395162843688593333333333333'), None, Decimal('0.4395162843688593333333333333')])

        print("✅ engine start")

    def test_signal(self):
        engine = Engine(
            dimensions_strategy="independent",
            dimensions_concurrency=3,
            random_seed=1
        )

        signal = engine._signal_()
        self.assertEqual(signal, [Decimal('0.13436424411240122'), Decimal('0.8474337369372327'), Decimal('0.763774618976614')])

        engine = Engine(dimensions_strategy="coupled", random_seed=1)
        signal = engine._signal_()

        self.assertEqual(signal, [Decimal('0.13436424411240122'), Decimal('0.8474337369372327'), Decimal('0.763774618976614')])

        print("✅ engine signal")

    def test_dimensions(self):
        random_seed = 1
        dimensions_concurrency = 5
        engine = Engine(
            random_seed=random_seed,
            dimensions_concurrency=dimensions_concurrency,
            dimensions_strategy="independent"
        )
        dimensions = engine._dimensions_()

        self.assertEqual(dimensions, SEMANTICS[:dimensions_concurrency])

        engine = Engine(
            random_seed=random_seed,
            dimensions_concurrency=dimensions_concurrency,
            dimensions_strategy="coupled"
        )
        dimensions = engine._dimensions_()

        self.assertEqual(dimensions, ['brand voice', 'artwork similarity', 'costume design', 'audience relevance', 'color scheme'])

    def test_responses(self):
        consensus = Consensus(
            strategy = "superposition",
            dimensions_strategy = "independent"
        )

        signal = [Decimal("0.6"), Decimal("0.5"), Decimal("0.1")]
        dimensions = ["illustration", "sentiment analysis", "artwork similiarity"]

        engine = Engine(random_seed=1)
        responses = engine._responses_(consensus=consensus, signal=signal, dimensions=dimensions)

        self.assertEqual(responses, {'The True Artist': [None, None, None, None, None, None, None, None, None, None, Decimal('-0.3'), None, None, None, None, None, None, None, None], 'The Executive Director': [None, None, None, None, None, None, None, None, None, None, Decimal('0.5'), None, None, None, None, None, None, None, None], 'The Technician': [None, None, None, None, None, None, None, None, None, None, Decimal('0.0'), None, None, None, None, None, None, None, None], 'The Critic': [None, None, None, None, None, None, None, None, None, None, Decimal('0.4'), None, None, None, None, None, None, None, None], 'The Fan': [None, None, None, None, None, None, None, None, None, None, Decimal('0.3'), None, None, None, None, None, None, None, None]})

if __name__ == '__main__':
    unittest.main()
