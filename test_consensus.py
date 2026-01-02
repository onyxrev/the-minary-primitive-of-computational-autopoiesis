#!/usr/bin/env python3

import os
import unittest
from decimal import Decimal
from engine import Iteration
from consensus import Consensus, PerspectiveSemanticAdjustment
from perspectives import Perspectives

STRATEGY_DEFAULT = "superposition"
DIMENSIONS_STRATEGY_DEFAULT = "independent"

class TestConsensus(unittest.TestCase):
    def test_initialization(self):
        strategy = "product"
        dimensions_strategy = "coupled"

        consensus = Consensus(strategy=strategy, dimensions_strategy=dimensions_strategy)
        self.assertEqual(consensus.strategy, strategy)
        self.assertEqual(consensus.dimensions_strategy, dimensions_strategy)

        print("✅ consensus initialization")

    def test_consensus(self):
        responses = {
            "The True Artist": [Decimal("0.1"), Decimal("0.5"), Decimal("0.9")],
            "The Executive Director": [Decimal("0.9"), Decimal("0.5"), Decimal("0.1")],
            "The Fan": [Decimal("0.5"), Decimal("0.1"), Decimal("0.9")]
        }

        consensus = Consensus(
            strategy="superposition",
            dimensions_strategy=DIMENSIONS_STRATEGY_DEFAULT
        )

        consensus_response, normalized_consensus_response = consensus.consensus(responses=responses)
        self.assertEqual(consensus_response, [Decimal('1.5'), Decimal('1.1'), Decimal('1.9')])
        self.assertEqual(normalized_consensus_response, [Decimal('0.5'), Decimal('0.3666666666666666666666666667'), Decimal('0.6333333333333333333333333333')])

        consensus = Consensus(
            strategy="product",
            dimensions_strategy=DIMENSIONS_STRATEGY_DEFAULT
        )

        consensus_response, normalized_consensus_response = consensus.consensus(responses=responses)
        self.assertEqual(consensus_response, [Decimal('0.391875'), Decimal('0.309375'), Decimal('0.496375')])
        self.assertEqual(normalized_consensus_response, consensus_response)

        print("✅ consensus calculation")

    def test_respond_for_perspective(self):
        perspective = Perspectives[0]

        # test historic adjustments get applied
        perspective.semantic_adjustments["information redaction"] = [
            PerspectiveSemanticAdjustment(iteration_id=0, semantic_name='information redaction', signal=Decimal('0.4'), adjustments_count=0, adjustments_sum=Decimal('0.4'), slow_exponential_moving_average=Decimal('0.400'), fast_exponential_moving_average=Decimal('0.40'))
        ]

        perspective.semantic_adjustments["color grading"] = [
            PerspectiveSemanticAdjustment(iteration_id=0, semantic_name='color grading', signal=Decimal('-0.1333333333333333333333333333'), adjustments_count=0, adjustments_sum=Decimal('-0.1333333333333333333333333333'), slow_exponential_moving_average=Decimal('-0.1333333333333333333333333333'), fast_exponential_moving_average=Decimal('-0.1333333333333333333333333333'))
        ]

        perspective.semantic_adjustments["physics"] = [
            PerspectiveSemanticAdjustment(iteration_id=0, semantic_name='physics', signal=Decimal('-0.2666666666666666666666666667'), adjustments_count=0, adjustments_sum=Decimal('-0.2666666666666666666666666667'), slow_exponential_moving_average=Decimal('-0.2666666666666666666666666667'), fast_exponential_moving_average=Decimal('-0.2666666666666666666666666667'))
        ]

        signal = [Decimal("0.8"), Decimal("0.5"), Decimal("0.2")]
        dimensions = ["information redaction", "color grading", "physics"]

        consensus = Consensus(
            strategy=STRATEGY_DEFAULT,
            dimensions_strategy="independent"
        )

        response = consensus.respond_for_perspective(
            perspective=perspective,
            signal=signal,
            dimensions=dimensions
        )

        self.assertEqual(
            response,
            [None, None, None, None, None, None, Decimal('-0.2333333333333333333333333333'), None, None, None, None, Decimal('1.000'), None, None, None, Decimal('-0.5666666666666666666666666667'), None, None, None]
        )

        consensus = Consensus(
            strategy=STRATEGY_DEFAULT,
            dimensions_strategy="coupled"
        )

        response = consensus.respond_for_perspective(
            perspective=perspective,
            signal=signal,
            dimensions=dimensions
        )

        # In coupled mode, same value is returned for all dimensions
        self.assertEqual(response, [None, None, None, None, None, None, Decimal('0.06666666666666666666666666667'), None, None, None, None, Decimal('0.06666666666666666666666666667'), None, None, None, Decimal('0.06666666666666666666666666667'), None, None, None])

        print("✅ consensus respond_for_perspective")

    def test_calculate_perspective_semantic_adjustments(self):
        perspective = Perspectives[0]
        response = [None, None, None, None, None, None, Decimal("0.1"), None, None, None, None, Decimal("0.5"), None, None, None, Decimal("0.9"), None, None, None]
        responses = {
            "The True Artist": response,
            "The Executive Director": [None, None, None, None, None, None, Decimal("0.9"), None, None, None, None, Decimal("0.5"), None, None, None, Decimal("0.1"), None, None, None],
            "The Fan": [None, None, None, None, None, None, Decimal("0.5"), None, None, None, None, Decimal("0.1"), None, None, None, Decimal("0.9"), None, None, None]
        }
        dimensions = list(perspective.semantic_profile.semantics.keys())
        signal = [Decimal("0.8"), Decimal("0.5"), Decimal("0.2")]
        consensus_answer = [None, None, None, None, None, None, Decimal('1.1'), None, None, None, None, Decimal('1.5'), None, None, None, Decimal('1.9'), None, None, None]
        normalized_consensus_answer = [None, None, None, None, None, None, Decimal('0.5'), None, None, None, None, Decimal('0.3666666666666666666666666667'), None, None, None, Decimal('0.6333333333333333333333333333'), None, None, None]

        iteration = Iteration(
            id = 0,
            responses = responses,
            signal = signal,
            consensus = consensus_answer,
            normalized_consensus = normalized_consensus_answer
        )

        consensus = Consensus(
            strategy=STRATEGY_DEFAULT,
            dimensions_strategy=DIMENSIONS_STRATEGY_DEFAULT
        )

        color_grading_adjustment, information_redaction_adjustment, physics_adjustment = consensus.calculate_perspective_semantic_adjustments(
            perspective=perspective,
            iteration=iteration,
            response=response,
            dimensions=dimensions,
            normalized_consensus=normalized_consensus_answer
        )

        self.assertTrue(
            color_grading_adjustment in perspective.semantic_adjustments["color grading"]
        )

        self.assertTrue(
            information_redaction_adjustment in perspective.semantic_adjustments["information redaction"]
        )

        self.assertTrue(
            physics_adjustment in perspective.semantic_adjustments["physics"]
        )

        print("✅ consensus calculate_perspective_semantic_adjustments")

if __name__ == '__main__':
    unittest.main()
