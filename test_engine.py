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
        self.assertEqual(engine.dimensions_concurrency, len(SEMANTICS))
        self.assertEqual(engine.dimensions_strategy, "independent")
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
        self.assertEqual(iterations[-1].signal, [Decimal('0.8755342442351592'), Decimal('0.30638662033324593'), Decimal('0.8585144063565593'), Decimal('0.31036362735313405'), Decimal('0.9392884321352825'), Decimal('0.7438421186671211'), Decimal('0.4161722627650255'), Decimal('0.25235810227983535'), Decimal('0.008480262463668842'), Decimal('0.8787178982088466'), Decimal('0.03791653059858058'), Decimal('0.8194141106127972'), Decimal('0.962201125180818'), Decimal('0.5702805702451802'), Decimal('0.17151709517771863'), Decimal('0.8677810644349934'), Decimal('0.9737752361596916'), Decimal('0.7040231423300713'), Decimal('0.5088737460778905')])
        self.assertEqual(iterations[-1].consensus, [Decimal('2.277671221175796000000000000'), Decimal('-0.3680668983337703500000000000'), Decimal('1.442572031782796500000000000'), Decimal('-1.648181863234329750000000000'), Decimal('1.656442160676412500000000000'), Decimal('1.219210593335605500000000000'), Decimal('-0.1691386861748725000000000000'), Decimal('-2.038209488600823250000000000'), Decimal('-2.457598687681655790000000000'), Decimal('1.393589491044233000000000000'), Decimal('-1.910417347007097100000000000'), Decimal('1.397070553063986000000000000'), Decimal('2.211005625904090000000000000'), Decimal('-0.6985971487740990000000000000'), Decimal('-2.092414524111406850000000000'), Decimal('2.188905322174967000000000000'), Decimal('1.948876180798458000000000000'), Decimal('0.02011571165035650000000000000'), Decimal('-0.0856312696105475000000000000')])

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
        self.assertEqual(signal, [Decimal('0.13436424411240122'), Decimal('0.8474337369372327'), Decimal('0.763774618976614'), Decimal('0.2550690257394217'), Decimal('0.49543508709194095'), Decimal('0.4494910647887381'), Decimal('0.651592972722763'), Decimal('0.7887233511355132'), Decimal('0.0938595867742349'), Decimal('0.02834747652200631'), Decimal('0.8357651039198697'), Decimal('0.43276706790505337'), Decimal('0.762280082457942'), Decimal('0.0021060533511106927'), Decimal('0.4453871940548014'), Decimal('0.7215400323407826'), Decimal('0.22876222127045265'), Decimal('0.9452706955539223'), Decimal('0.9014274576114836')])

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
