"""
Consensus formation and response calculation for the Minary framework.

This module implements the core mathematical operations from
"The Minary Primitive of Computational Autopoiesis" (Connor & Defant, 2026).

Key operations and their paper references:
    - Raw response: r(t)_i,j = x(t)_j - C_i,j          (Equation 1)
    - Adjusted response: R(t)_i,j = r(t)_i,j + Δ(t-1)  (Equation 2)
    - Average adjusted response: R(t)_i = (1/k)Σ R(t)_i,j  (Equation 3)
    - Consensus: G(t) = Σ_i R(t)_i                     (Equation 4)
    - Learning signal: d(t)_i = G̅(t) - R(t)_i         (Equation 6)

The critical autopoietic property (Equation 8): the input signal x(t)
cancels out of the learning signal d(t), making the system self-referential.
"""

from math import prod
from typing import List, Dict, Tuple
from decimal import Decimal
from data_types import Iteration, Perspective, PerspectiveSemanticAdjustment

POSITIVE_ONE = Decimal("1")
POSITIVE_ZERO = Decimal("0")
NEGATIVE_ONE = Decimal("-1")

class Consensus:
    """
    Orchestrates response generation and consensus formation.

    This class implements both the perspective response logic (Equations 1-3)
    and the consensus aggregation (Equation 4), as well as the EMA update
    logic that computes learning signals (Equation 6).

    Two consensus strategies are supported:
        - "superposition": Linear sum (the Minary approach, preserves information)
        - "product": Multiplicative (traditional Bayesian, collapses information)

    Two dimension strategies are supported:
        - "independent": Each dimension processed separately
        - "coupled": Responses averaged across active dimensions (Equation 3)

    Attributes:
        strategy: "superposition" or "product"
        dimensions_strategy: "independent" or "coupled"
    """

    def __init__(self, strategy: str, dimensions_strategy: str):
        self.strategy = strategy
        self.dimensions_strategy = dimensions_strategy

    def consensus(self, responses: Dict[str, List[Decimal]]):
        """
        Aggregate perspective responses into collective consensus.

        Paper Reference: Equation 4
            G(t) = Σ_i R(t)_i

        Args:
            responses: Dict mapping perspective name to response vector R(t)_i

        Returns:
            [consensus, normalized_consensus] where:
                - consensus: G(t) raw aggregation
                - normalized_consensus: G̅(t) = G(t)/n
        """
        consensus_response = None
        normalized_consensus_response = None

        if self.strategy == "product":
            consensus_response = self._product_(responses)
            normalized_consensus_response = consensus_response
        elif self.strategy == "superposition":
            consensus_response = self._superposition_(responses)
            normalized_consensus_response = self._normalized_consensus_(consensus_response, responses)
        else:
            raise ValueError("invalid consensus strategy")

        return [consensus_response, normalized_consensus_response]

    def _product_(self, responses: Dict[str, List[Decimal]]):
        """
        Multiplicative consensus (traditional Bayesian approach).

        Transforms responses from [-1, 1] to [0, 1], then multiplies.
        Included for comparison; NOT the Minary approach.

        Note: Product consensus collapses toward zero and amplifies noise.
        This is the allopoietic behavior Minary is designed to avoid.

        Args:
            responses: Dict mapping perspective name to response vector

        Returns:
            Product consensus vector (already normalized)
        """
        values_list = list(responses.values())

        if not values_list:
            return []

        # Transform from [-1, 1] to [0, 1] probability range
        probability_range_values = [
            [
                (component / 2) + Decimal("0.5") if component is not None else None
                for component in components
            ]
            for components in values_list
        ]

        consensus_by_dimension = [
            prod(dimension_values) if dimension_values[0] is not None else None
            for dimension_values in zip(*probability_range_values)
        ]

        return consensus_by_dimension

    def _superposition_(self, responses: List[Decimal]):
        """
        Linear superposition consensus (the Minary approach).

        Paper Reference: Equation 4
            G(t) = Σ_i R(t)_i

        Simply sums all perspective responses. This preserves information
        through constructive/destructive interference rather than collapsing
        it through multiplication.

        Args:
            responses: Dict mapping perspective name to response vector R(t)_i

        Returns:
            Raw consensus vector G(t) (sum of all responses)
        """
        return [
            sum(column) if column[0] is not None else None
            for column in zip(*responses.values())
        ]

    def _normalized_consensus_(
            self,
            consensus_response: List[Decimal],
            responses: Dict[str, List]
    ):
        """
        Normalize consensus by number of perspectives.

        Paper Reference: Used in Equation 6
            G̅(t) = G(t) / n

        Args:
            consensus_response: Raw consensus G(t)
            responses: Dict of responses (used to count perspectives n)

        Returns:
            Normalized consensus G̅(t)
        """
        return [
            component / Decimal(str(len(responses.keys()))) if component is not None else None
            for component in consensus_response
        ]

    def respond_for_perspective(
            self,
            perspective: Perspective,
            signal: List[Decimal],
            dimensions: List[str]
    ):
        """
        Generate a perspective's response to the input signal.

        Dispatches to either independent or coupled response strategy.

        Args:
            perspective: The perspective p_i generating the response
            signal: Input vector x(t) with k components
            dimensions: List of k active dimension names S(t)

        Returns:
            Response vector R(t)_i with m components (None for inactive dimensions)
        """
        if self.dimensions_strategy == "independent":
            return self._respond_independent_(perspective, signal, dimensions)
        elif self.dimensions_strategy == "coupled":
            return self._respond_coupled_(perspective, signal, dimensions)
        else:
            return None

    def _respond_independent_(
            self,
            perspective: Perspective,
            signal: List[Decimal],
            dimensions: List[str]
    ):
        """
        Generate response with dimensions processed independently.

        Each dimension gets its own adjusted response R(t)_i,j without
        cross-dimensional averaging. Simpler dynamics, orthogonal dimensions.

        Paper Reference: Equations 1-2 (without Equation 3 averaging)

        Args:
            perspective: The perspective p_i
            signal: Input vector x(t)
            dimensions: Active dimensions S(t)

        Returns:
            Response vector with independent per-dimension values
        """
        # Get all m possible semantic dimensions
        all_possible_semantic_dimensions = list(perspective.semantic_profile.semantics.keys())

        # Create a full-sized output vector (m dimensions, None for inactive)
        output = [None] * len(all_possible_semantic_dimensions)

        all_semantic_pairs = dict(perspective.semantic_profile.semantics.items())

        # Iterate through the k ACTIVE dimensions
        for i, semantic_name in enumerate(dimensions):
            if semantic_name in all_semantic_pairs:
                semantic_pair = (semantic_name, all_semantic_pairs[semantic_name])

                # Calculate adjusted response: R(t)_i,j = r(t)_i,j + Δ(t-1)_i,j
                value = self._adjusted_response_for_semantic_(
                    perspective=perspective,
                    signal_component=signal[i],
                    semantic=semantic_pair
                )

                # Place in correct position in full m-dimensional vector
                dimension_index = all_possible_semantic_dimensions.index(semantic_name)
                output[dimension_index] = value

        return output

    def _respond_coupled_(
            self,
            perspective: Perspective,
            signal: List[Decimal],
            dimensions: List[str]
    ):
        """
        Generate response with dimensions coupled via averaging.

        Computes adjusted response for each active dimension, then averages
        them to produce a single value used for all active dimensions.
        This creates cross-dimensional interactions.

        Paper Reference: Equation 3
            R(t)_i = (1/k) Σ_{j∈S(t)} R(t)_i,j

        This is the mode used in the paper's worked examples (Section 7).
        Enables emergent phenomena like the halo effect.

        Args:
            perspective: The perspective p_i
            signal: Input vector x(t)
            dimensions: Active dimensions S(t)

        Returns:
            Response vector where all active dimensions have the same averaged value
        """
        all_possible_semantic_dimensions = list(perspective.semantic_profile.semantics.keys())
        output = [None] * len(all_possible_semantic_dimensions)
        raw_adjusted_responses = []

        # Gather competencies and EMA values for active dimensions
        competencies = []
        differentials = []
        for semantic_name in dimensions:
            competencies.append(perspective.semantic_profile.semantics[semantic_name])
            differentials.append(self._differential_for_semantic_(perspective, semantic_name))

        # Compute adjusted response for each active dimension
        for i, semantic_name in enumerate(all_possible_semantic_dimensions):
            if semantic_name in dimensions:
                dimension_index = dimensions.index(semantic_name)
                # R(t)_i,j = r(t)_i,j + Δ(t-1)_i,j
                value = self._raw_response_for_semantic_(
                    signal_component=signal[dimension_index],
                    competency=competencies[dimension_index]
                ) + differentials[dimension_index]
                output[i] = value
                raw_adjusted_responses.append(value)

        # Equation 3: Average across all active dimensions
        avg = sum(raw_adjusted_responses) / len(raw_adjusted_responses)

        # Apply same averaged value to all active dimensions (coupled behavior)
        for i, semantic_name in enumerate(all_possible_semantic_dimensions):
            if output[i] is not None:
                output[i] = avg

        return output

    def _raw_response_for_semantic_(
            self,
            signal_component: Decimal,
            competency: Decimal
    ):
        """
        Compute raw response: signal minus competency.

        Paper Reference: Equation 1
            r(t)_i,j = x(t)_j - C_i,j

        A perspective with high competency (close to 1) will have a more
        negative raw response. A perspective with low competency (close to 0)
        will have a more positive raw response.

        Args:
            signal_component: Input signal x(t)_j for this dimension
            competency: Perspective's competency C_i,j for this dimension

        Returns:
            Raw response r(t)_i,j in range approximately [-1, 1]
        """
        return signal_component - competency

    def _differential_for_semantic_(
            self,
            perspective: Perspective,
            semantic_name: str
    ):
        """
        Get the current EMA adjustment Δ(t-1)_i,j for a perspective-dimension pair.

        This is the "memory" that makes the system autopoietic. It accumulates
        learning over time and is added to raw responses.

        Args:
            perspective: The perspective p_i
            semantic_name: The dimension s_j

        Returns:
            Current Δ(t-1)_i,j value, or 0 if dimension never active
        """
        latest_adjustment = perspective.get_current_adjustment_for_semantic(semantic_name)

        if latest_adjustment:
            return latest_adjustment.slow_exponential_moving_average
        else:
            return POSITIVE_ZERO

    def _adjusted_response_for_semantic_(
            self,
            perspective: Perspective,
            signal_component: Decimal,
            semantic: Tuple
    ):
        """
        Compute adjusted response: raw response plus EMA memory.

        Paper Reference: Equation 2
            R(t)_i,j = r(t)_i,j + Δ(t-1)_i,j

        The adjusted response incorporates both the immediate signal-competency
        difference AND the accumulated learning from past iterations.

        Args:
            perspective: The perspective p_i
            signal_component: Input signal x(t)_j
            semantic: Tuple of (dimension_name, competency_value)

        Returns:
            Adjusted response R(t)_i,j
        """
        semantic_name, competency = semantic

        raw_response = self._raw_response_for_semantic_(signal_component, competency)
        differential = self._differential_for_semantic_(perspective, semantic_name)
        return raw_response + differential

    def _adjustment_differential_(
            self,
            i: int,
            normalized_consensus: List[Decimal],
            response: List[Decimal]
    ):
        """
        Compute learning signal: normalized consensus minus individual response.

        Paper Reference: Equation 6
            d(t)_i = (1/n)G(t) - R(t)_i = G̅(t) - R(t)_i

        CRITICAL PROPERTY (Equation 8): When you sum all learning signals,
        they equal zero: Σ_i d(t)_i = 0. This makes the system zero-sum
        and self-referential. The input signal x(t) cancels out completely.

        Args:
            i: Dimension index j
            normalized_consensus: G̅(t) vector
            response: This perspective's response R(t)_i vector

        Returns:
            Learning signal d(t)_i for dimension j
        """
        return normalized_consensus[i] - response[i]

    def calculate_perspective_semantic_adjustments(
            self,
            perspective: Perspective,
            iteration: Iteration,
            response: List[Decimal],
            dimensions: List[str],
            normalized_consensus: List[Decimal]
    ):
        """
        Update a perspective's EMA memory based on this iteration's learning signal.

        This is where the autopoietic feedback loop closes. For each active
        dimension, we:
        1. Compute learning signal d(t)_i = G̅(t) - R(t)_i  (Equation 6)
        2. Update EMA: Δ(t)_i,j = α·d(t)_i + (1-α)·Δ(t-1)_i,j  (Equation 5)

        Paper Reference: Section 4, Equations 5-6
            The key insight is that d(t) depends only on competency differences,
            not on the input signal x(t) (see Equation 8).

        Args:
            perspective: The perspective p_i to update
            iteration: Current iteration data (for logging)
            response: This perspective's response R(t)_i
            dimensions: Active dimensions S(t)
            normalized_consensus: G̅(t) vector

        Returns:
            List of new PerspectiveSemanticAdjustment objects created
        """
        all_possible_semantic_dimensions = list(perspective.semantic_profile.semantics.keys())
        semantic_names = dimensions

        all_adjustments_list = []
        for i in range(len(dimensions)):
            semantic_name = semantic_names[i]
            dimension_index = all_possible_semantic_dimensions.index(semantic_name)

            if normalized_consensus[dimension_index] is None:
                continue

            # Equation 6: d(t)_i = G̅(t) - R(t)_i
            differential = self._adjustment_differential_(
                dimension_index,
                normalized_consensus,
                response
            )

            adjustments_list = perspective.semantic_adjustments.get(semantic_name, [])
            current_adjustment = adjustments_list[-1] if adjustments_list else None

            new_adjustment = PerspectiveSemanticAdjustment(
                iteration_id=iteration.id,
                semantic_name=semantic_name,
                signal=differential,  # The learning signal d(t)_i
                dimension_index=dimension_index,
                adjustments_count=current_adjustment.adjustments_count + 1 if current_adjustment else 1,
                adjustments_sum=current_adjustment.adjustments_sum + differential if current_adjustment else differential
            )

            # Equation 5: Update EMA with new learning signal
            new_adjustment.populate_exponential_moving_averages(differential, current_adjustment)
            adjustments_list.append(new_adjustment)
            all_adjustments_list.append(new_adjustment)
            perspective.semantic_adjustments[semantic_name] = adjustments_list

        return all_adjustments_list
