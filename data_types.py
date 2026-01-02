"""
Core data structures for the Minary autopoietic simulation.

This module defines the fundamental types that implement the mathematical
formalism from "The Minary Primitive of Computational Autopoiesis"
(Connor & Defant, 2026).

Key mappings to paper notation:
    - Perspective: A single evaluator p_i with competencies C_i,j
    - SemanticProfile: The competency row C_i,· for perspective i
    - PerspectiveSemanticAdjustment: The EMA memory Δ(t)_i,j
    - Iteration: One time step t containing signal x(t), responses R(t), consensus G(t)
"""

from typing import List, Dict
from dataclasses import dataclass
from decimal import Decimal

# Step size α for EMA updates (Paper Section 4, Equation 5)
# Paper assumes α ∈ (0, 2/3) for convergence proofs
SLOW_EMA_ALPHA = Decimal("0.02")  # Primary adaptation mechanism
FAST_EMA_ALPHA = Decimal("0.1")   # For trend analysis (not used in core math)

@dataclass
class SemanticProfile:
    """
    The competency profile for a single perspective.

    Represents one row of the competency matrix C from Paper Section 4.
    Maps semantic dimension names s_j to competency values C_i,j ∈ [0, 1].

    Attributes:
        semantics: Dict mapping dimension name to competency value.
                   Example: {"modern art": Decimal("0.95"), "physics": Decimal("0.5")}
    """
    semantics: Dict[str, Decimal]


@dataclass
class PerspectiveSemanticAdjustment:
    """
    The EMA memory state Δ(t)_i,j for one perspective-dimension pair.

    This is the core "identity" structure of the autopoietic system.
    The slow_exponential_moving_average field holds the value that gets
    added to raw responses to form adjusted responses (Paper Equation 2).

    Paper Reference: Section 4, Equations 5-6
        Δ(t)_i,j = α·d(t)_i + (1-α)·Δ(t-1)_i,j  (for active dimensions)

    Attributes:
        iteration_id: The time step t when this adjustment was computed
        semantic_name: The dimension s_j this adjustment applies to
        signal: The learning signal d(t)_i = G̅(t) - R(t)_i (Equation 6)
        adjustments_count: How many times this dimension has been active
        adjustments_sum: Cumulative sum of signals (for analysis)
        dimension_index: Index j in the full m-dimensional output vector
        slow_exponential_moving_average: The actual Δ(t)_i,j value (α=0.02)
        fast_exponential_moving_average: Faster EMA for trend analysis (α=0.1)
    """
    iteration_id: int
    semantic_name: str
    signal: Decimal
    adjustments_count: int
    adjustments_sum: Decimal
    dimension_index: int = None
    slow_exponential_moving_average: Decimal = None
    fast_exponential_moving_average: Decimal = None

    def calculate_exponential_moving_average(self, old, new, alpha):
        """
        Compute EMA update: new_ema = α·new + (1-α)·old

        Paper Reference: Equation 5
            Δ(t)_i,j = α·d(t)_i + (1-α)·Δ(t-1)_i,j

        Args:
            old: Previous EMA value Δ(t-1)
            new: New signal value d(t)
            alpha: Step size α ∈ (0, 2/3)

        Returns:
            Updated EMA value Δ(t)
        """
        return (new * alpha) + (old * (1 - alpha))

    def populate_exponential_moving_averages(self, differential, old=None):
        """
        Update both slow and fast EMAs with a new learning signal.

        Args:
            differential: The learning signal d(t)_i from Equation 6
            old: Previous PerspectiveSemanticAdjustment (None if first iteration)
        """
        old_slow_ema = old.slow_exponential_moving_average if old else differential
        old_fast_ema = old.fast_exponential_moving_average if old else differential

        self.slow_exponential_moving_average = self.calculate_exponential_moving_average(
            old_slow_ema,
            differential,
            SLOW_EMA_ALPHA
        )

        self.fast_exponential_moving_average = self.calculate_exponential_moving_average(
            old_fast_ema,
            differential,
            FAST_EMA_ALPHA
        )

@dataclass
class Iteration:
    """
    A single time step t in the simulation.

    Captures all inputs and outputs for one iteration of the Minary process.

    Paper Reference: Section 4
        - signal: The random input x(t) sampled from distribution μ
        - responses: The adjusted responses R(t)_i for each perspective
        - consensus: The raw consensus G(t) = Σ_i R(t)_i (Equation 4)
        - normalized_consensus: G̅(t) = G(t)/n

    Attributes:
        id: Time step index t
        signal: Input vector x(t) with k components (one per active dimension)
        responses: Dict mapping perspective name to response vector R(t)_i
        consensus: Raw superposition G(t) (or product consensus)
        normalized_consensus: G̅(t) = G(t)/n, used for learning signals
    """
    id: int
    signal: List[Decimal]
    responses: Dict[str, List[Decimal]]
    consensus: List[Decimal]
    normalized_consensus: List[Decimal]


@dataclass
class Perspective:
    """
    A single perspective p_i in the Minary system.

    Represents one of the n evaluators that contribute to consensus.
    Each perspective has fixed competencies (semantic_profile) and
    evolving learned adjustments (semantic_adjustments).

    Paper Reference: Section 4
        - semantic_profile.semantics[j] = C_i,j (competency matrix row)
        - semantic_adjustments[j][-1] = Δ(t)_i,j (current EMA state)

    Attributes:
        name: Human-readable identifier (e.g., "The True Artist")
        semantic_profile: The competency values C_i,· for this perspective
        semantic_adjustments: Dict mapping dimension name to list of
                              PerspectiveSemanticAdjustment history
    """
    name: str
    semantic_profile: SemanticProfile
    semantic_adjustments: Dict[str, List[PerspectiveSemanticAdjustment]]

    def get_current_adjustment_for_semantic(self, semantic_name: str):
        """
        Get the most recent EMA state Δ(t)_i,j for a given dimension.

        Args:
            semantic_name: The dimension s_j to look up

        Returns:
            The latest PerspectiveSemanticAdjustment, or None if dimension
            has never been active for this perspective.
        """
        adjustments_list = self.semantic_adjustments.get(semantic_name, [])
        return adjustments_list[-1] if adjustments_list else None
