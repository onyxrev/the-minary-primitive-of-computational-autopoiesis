"""
Simulation engine for the Minary autopoietic framework.

This module orchestrates the discrete-time stochastic simulation described
in "The Minary Primitive of Computational Autopoiesis" (Connor & Defant, 2026).

Each iteration of the simulation:
    1. Samples a random signal x(t) from uniform distribution on [0,1]
    2. Selects k active dimensions S(t) (randomly if coupled, fixed if independent)
    3. Each perspective generates a response R(t)_i
    4. Responses are combined via superposition to form consensus G(t)
    5. Learning signals d(t)_i are computed and EMA memory Δ(t) is updated

Paper Reference: Section 4 (Mathematical Formalism) and Section 7 (Worked Example)
"""

import random
from decimal import Decimal
from typing import Optional, List, Dict
from data_types import Iteration
from consensus import Consensus
from perspectives import TheSignal, Perspectives
from persistence import PersistenceManager
from csv_export import CSVExporter

# Default configuration
CONSENSUS_STRATEGY_DEFAULT = "superposition"  # Linear sum (Minary approach)
SEMANTICS = list(Perspectives[0].semantic_profile.semantics.keys())  # All m=19 dimensions
DIMENSIONS_CONCURRENCY_DEFAULT = 3  # k = number of active dimensions per iteration
DIMENSIONS_CONCURRENCY_MIN = 1
DIMENSIONS_CONCURRENCY_MAX = len(SEMANTICS)
DIMENSIONS_STRATEGY_DEFAULT = "coupled"  # vs "independent"
CSV_EXPORT_ENABLED_DEFAULT = False
PERSPECTIVES_COUNT_MIN = 1
PERSPECTIVES_COUNT_MAX = len(Perspectives)  # n = number of perspectives
PERSPECTIVES_COUNT_DEFAULT = PERSPECTIVES_COUNT_MAX
RANDOM_SEED_DEFAULT = None
ITERATIONS_COUNT_MAX = 100_000

class Engine:
    """
    Main simulation engine for the Minary autopoietic system.

    Runs the discrete-time stochastic process described in the paper,
    managing perspectives, generating signals, computing consensus,
    and updating the EMA memory matrices.

    Paper Reference: Section 4
        - n perspectives: p_1, ..., p_n
        - m semantic dimensions: s_1, ..., s_m
        - k active dimensions per iteration: S(t) ⊆ [m]
        - Signal x(t) sampled from distribution μ (uniform [0,1])

    Attributes:
        consensus_strategy: "superposition" (Minary) or "product" (Bayesian)
        dimensions_concurrency: k, number of active dimensions per iteration
        dimensions_strategy: "independent" or "coupled"
        perspectives: List of n Perspective objects
        iteration_count: Current time step t
        iterations: History of Iteration objects
    """

    def __init__(
        self,
        persistence_manager: PersistenceManager = None,
        consensus_strategy: str = CONSENSUS_STRATEGY_DEFAULT,
        dimensions_concurrency: int = DIMENSIONS_CONCURRENCY_DEFAULT,
        dimensions_strategy: str = DIMENSIONS_STRATEGY_DEFAULT,
        csv_export_enabled: bool = CSV_EXPORT_ENABLED_DEFAULT,
        perspectives_count: int = PERSPECTIVES_COUNT_DEFAULT,
        random_seed: int = RANDOM_SEED_DEFAULT,
        iterations_count_max: int = ITERATIONS_COUNT_MAX
    ):
        self.consensus_strategy = consensus_strategy
        self.dimensions_concurrency = max(DIMENSIONS_CONCURRENCY_MIN, min(dimensions_concurrency, DIMENSIONS_CONCURRENCY_MAX))
        self.dimensions_strategy = dimensions_strategy
        self.csv_export_enabled = csv_export_enabled
        self.perspectives_count = max(PERSPECTIVES_COUNT_MIN, min(perspectives_count, PERSPECTIVES_COUNT_MAX))
        self.persistence_manager = persistence_manager

        self.random_seed = random_seed
        if random_seed:
            random.seed(random_seed)

        self.iterations_count_max = iterations_count_max

        self.csv_exporter = None
        if self.csv_export_enabled:
            self.csv_exporter = CSVExporter()

        self.signal_perspective = TheSignal
        self.perspectives = Perspectives[:self.perspectives_count]
        self.iteration_count = 0
        self.iterations = []

    @classmethod
    def load_from_persistence(
        cls,
        persistence_manager: PersistenceManager | None,
        perspectives_count: int = PERSPECTIVES_COUNT_DEFAULT,
        consensus_strategy: str = CONSENSUS_STRATEGY_DEFAULT,
        dimensions_concurrency: int = DIMENSIONS_CONCURRENCY_DEFAULT,
        dimensions_strategy: str = DIMENSIONS_STRATEGY_DEFAULT,
        csv_export_enabled: bool = CSV_EXPORT_ENABLED_DEFAULT,
        random_seed: int = RANDOM_SEED_DEFAULT,
        iterations_count_max: int = ITERATIONS_COUNT_MAX
    ):
        """
        Create an Engine, optionally restoring from saved state.

        Allows resuming a simulation from a checkpoint. If no saved state
        exists, creates a fresh Engine with the specified parameters.

        Args:
            persistence_manager: Optional manager for state save/load
            perspectives_count: n, number of perspectives to use
            consensus_strategy: "superposition" or "product"
            dimensions_concurrency: k, number of active dimensions
            dimensions_strategy: "independent" or "coupled"
            csv_export_enabled: Whether to export data to CSV
            random_seed: Seed for reproducibility (None for random)
            iterations_count_max: Total iterations to run

        Returns:
            Engine instance (fresh or restored from persistence)
        """
        state = persistence_manager.load_state() if persistence_manager else None
        if state is None:
            engine = cls(
                consensus_strategy=consensus_strategy,
                dimensions_concurrency=dimensions_concurrency,
                dimensions_strategy=dimensions_strategy,
                csv_export_enabled=csv_export_enabled,
                perspectives_count=perspectives_count,
                random_seed=random_seed,
                iterations_count_max=iterations_count_max
            )
            engine.persistence_manager = persistence_manager
            return engine

        perspectives, iterations, iteration_count = state

        engine = cls(
            consensus_strategy=consensus_strategy,
            dimensions_concurrency=dimensions_concurrency,
            dimensions_strategy=dimensions_strategy,
            csv_export_enabled=csv_export_enabled,
            perspectives_count=len(perspectives),
            random_seed=random_seed,
            iterations_count_max=iterations_count_max
        )

        engine.persistence_manager = persistence_manager
        engine.perspectives = perspectives
        engine.iterations = iterations
        engine.iteration_count = iteration_count

        return engine

    def start(self):
        """
        Run the simulation for iterations_count_max iterations.

        This is the main loop that drives the autopoietic process.
        Each iteration follows the procedure in Paper Section 4.

        Returns:
            List of Iteration objects from the simulation
        """
        consensus = Consensus(
            strategy=self.consensus_strategy,
            dimensions_strategy=self.dimensions_strategy
        )

        while self.iteration_count < self.iterations_count_max:
            print(f"iteration: {self.iteration_count}")

            iteration, dimensions = self.iterate(consensus)
            self.iterations.append(iteration)
            self.iteration_count = self.iteration_count + 1

            # CSV export for analysis
            if self.csv_exporter:
                self.csv_exporter.log_iteration(iteration)
                for perspective in self.perspectives:
                    self.csv_exporter.log_semantic_adjustments(
                        iteration.id,
                        perspective.name,
                        perspective.semantic_adjustments,
                        dimensions
                    )

            if self.persistence_manager and self.persistence_manager.should_fsync(self.iteration_count):
                self.persistence_manager.save_state(self.perspectives, self.iterations, self.iteration_count)
                self.iterations = self.iterations[-self.persistence_manager.fsync_interval:]

        return self.iterations

    def _signal_(self):
        """
        Generate random input signal x(t).

        Paper Reference: Section 4
            "we sample a random signal x(t)_j ∈ [0, 1] from the probability
            distribution μ" (uniform distribution in this implementation)

        Returns:
            List of k Decimal values, each uniformly distributed in [0, 1]
        """
        return [Decimal(str(random.random())) for _n in range(self.dimensions_concurrency)]

    def _dimensions_(self):
        """
        Select which k dimensions are active this iteration.

        Paper Reference: Section 4
            "At each time step t, we choose a k-element set S(t) ⊆ [m]
            uniformly at random"

        For "independent" strategy: always uses first k dimensions (deterministic)
        For "coupled" strategy: randomly samples k dimensions (stochastic)

        Returns:
            List of k dimension names (the active set S(t))
        """
        if self.dimensions_strategy == "independent":
            return SEMANTICS[:self.dimensions_concurrency]
        elif self.dimensions_strategy == "coupled":
            return random.sample(SEMANTICS, self.dimensions_concurrency)
        else:
            return None

    def _responses_(self, consensus: Consensus, signal: List[Decimal], dimensions: List[str]):
        """
        Collect responses from all perspectives.

        Each perspective generates its adjusted response R(t)_i based on
        the signal and its competencies/EMA state.

        Args:
            consensus: Consensus object that computes responses
            signal: Input vector x(t)
            dimensions: Active dimensions S(t)

        Returns:
            Dict mapping perspective name to response vector R(t)_i
        """
        return {
            perspective.name: consensus.respond_for_perspective(
                perspective,
                signal,
                dimensions
            )
            for perspective in self.perspectives
        }

    def iterate(self, consensus: Consensus):
        """
        Execute one iteration of the Minary process.

        This implements the full iteration cycle from Paper Section 4:
            1. Sample signal x(t) and select active dimensions S(t)
            2. Each perspective computes adjusted response R(t)_i
            3. Aggregate via superposition: G(t) = Σ R(t)_i
            4. Compute learning signals: d(t)_i = G̅(t) - R(t)_i
            5. Update EMA: Δ(t) = α·d(t) + (1-α)·Δ(t-1)

        Args:
            consensus: Consensus object for response/aggregation logic

        Returns:
            Tuple of (Iteration object, list of active dimension names)
        """
        # Step 1: Sample signal and select dimensions
        signal = self._signal_()
        dimensions = self._dimensions_()

        # Step 2: Collect all perspective responses
        responses = self._responses_(consensus, signal, dimensions)

        # Step 3: Aggregate via superposition (or product)
        consensus_answer, normalized_consensus_answer = consensus.consensus(responses)

        iteration = Iteration(
            id=self.iteration_count,
            responses=responses,
            signal=signal,
            consensus=consensus_answer,
            normalized_consensus=normalized_consensus_answer
        )

        # Steps 4-5: Compute learning signals and update EMA for each perspective
        for perspective in self.perspectives:
            consensus.calculate_perspective_semantic_adjustments(
                perspective=perspective,
                iteration=iteration,
                response=responses.get(perspective.name),
                dimensions=dimensions,
                normalized_consensus=normalized_consensus_answer
            )

        return iteration, dimensions
