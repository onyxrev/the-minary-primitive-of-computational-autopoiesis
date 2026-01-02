import csv
import os
from typing import List, Dict
from data_types import Iteration, Perspective, PerspectiveSemanticAdjustment

class CSVExporter:
    def __init__(self, base_path: str = "export"):
        self.base_path = base_path
        self.iterations_file = os.path.join(base_path, "iterations.csv")
        self.adjustments_file = os.path.join(base_path, "semantic_adjustments.csv")

        # Create export directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        # Initialize iterations.csv header if file doesn't exist
        if not os.path.exists(self.iterations_file):
            # We'll write the header when we get the first iteration since we need to know signal dimensions
            pass

        # Initialize semantic_adjustments.csv header if file doesn't exist
        if not os.path.exists(self.adjustments_file):
            with open(self.adjustments_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration_id', 'perspective_name', 'semantic_name', 'dimension_position', 'signal',
                    'adjustments_count', 'adjustments_sum', 'slow_ema', 'fast_ema'
                ])

    def _write_iterations_header(self, iteration: Iteration):
        """Write header for iterations.csv based on signal dimensions"""
        if os.path.exists(self.iterations_file):
            return  # Header already exists

        header = ['iteration_id']

        # Signal columns
        for i in range(len(iteration.signal)):
            header.append(f'signal_{i}')

        # Consensus columns
        for i in range(len(iteration.consensus)):
            header.append(f'consensus_{i}')

        # Normalized consensus columns
        for i in range(len(iteration.normalized_consensus)):
            header.append(f'normalized_consensus_{i}')

        with open(self.iterations_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_iteration(self, iteration: Iteration):
        """Append iteration data to iterations.csv"""
        # Write header if this is the first iteration
        if len(iteration.signal) > 0:
            self._write_iterations_header(iteration)

        # Prepare row data
        row = [iteration.id]

        # Add signal values
        row.extend(
            [
                str(val) if val is not None else ""
                for val in iteration.signal
            ]
        )

        # Add consensus values
        row.extend(
            [
                str(val) if val is not None else ""
                for val in iteration.consensus
            ]
        )

        # Add normalized consensus values
        row.extend(
            [
                str(val) if val is not None else ""
                for val in iteration.normalized_consensus
            ]
        )

        # Append to file
        with open(self.iterations_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_semantic_adjustments(self, iteration_id: int, perspective_name: str,
                                adjustments: Dict[str, List[PerspectiveSemanticAdjustment]],
                                current_dimensions: List[str] = None):
        """Append semantic adjustment data to semantic_adjustments.csv"""
        rows = []

        # If current_dimensions is provided, only log those dimensions
        # Otherwise, log all dimensions (backward compatibility)
        if current_dimensions is not None:
            dimensions_to_log = list(enumerate(current_dimensions))
        else:
            # Backward compatibility: no position info available
            dimensions_to_log = [(None, semantic_name) for semantic_name in adjustments.keys()]

        for position, semantic_name in dimensions_to_log:
            if semantic_name in adjustments and adjustments[semantic_name]:
                adjustment_list = adjustments[semantic_name]
                # Get the most recent adjustment for this iteration
                latest_adjustment = adjustment_list[-1]

                # Only log if this adjustment is from the current iteration
                if latest_adjustment.iteration_id == iteration_id:
                    row = [
                        iteration_id,
                        perspective_name,
                        semantic_name,
                        latest_adjustment.dimension_index,
                        str(latest_adjustment.signal),
                        latest_adjustment.adjustments_count,
                        str(latest_adjustment.adjustments_sum),
                        str(latest_adjustment.slow_exponential_moving_average) if latest_adjustment.slow_exponential_moving_average is not None else '',
                        str(latest_adjustment.fast_exponential_moving_average) if latest_adjustment.fast_exponential_moving_average is not None else ''
                    ]
                    rows.append(row)

        # Append all rows to file
        if rows:
            with open(self.adjustments_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    def export_all(self, perspectives: List[Perspective], iterations: List[Iteration]):
        """One-time export of all current engine state"""
        # Export all iterations
        for iteration in iterations:
            self.log_iteration(iteration)

        # Export all semantic adjustments (use None for backward compatibility - exports all dimensions)
        for iteration in iterations:
            for perspective in perspectives:
                self.log_semantic_adjustments(
                    iteration.id,
                    perspective.name,
                    perspective.semantic_adjustments,
                    None  # Export all dimensions for historical data
                )
