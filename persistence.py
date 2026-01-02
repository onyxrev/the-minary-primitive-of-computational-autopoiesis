import pickle
import os
from typing import List, Dict, Optional, Tuple
from data_types import Iteration, Perspective

class PersistenceManager:
    def __init__(self, persistence_path: str = "autopoetic_state.pkl", fsync_interval: int = 10000):
        self.persistence_path = persistence_path
        self.fsync_interval = fsync_interval

    def save_state(self, perspectives: List[Perspective], iterations: List[Iteration], iteration_count: int):
        # Truncate semantic adjustments to keep only the latest record
        truncated_perspectives = []
        for perspective in perspectives:
            truncated_adjustments = {}
            for semantic_name, adjustments_list in perspective.semantic_adjustments.items():
                if adjustments_list:
                    truncated_adjustments[semantic_name] = [adjustments_list[-1]]
                else:
                    truncated_adjustments[semantic_name] = []

            # Create a copy of the perspective with truncated adjustments
            truncated_perspective = Perspective(
                name=perspective.name,
                semantic_profile=perspective.semantic_profile,
                semantic_adjustments=truncated_adjustments
            )
            truncated_perspectives.append(truncated_perspective)

        # Keep only recent iterations (last fsync_interval worth)
        recent_iterations = iterations[-self.fsync_interval:] if len(iterations) > self.fsync_interval else iterations

        state = {
            "perspectives": truncated_perspectives,
            "iterations": recent_iterations,
            "iteration_count": iteration_count
        }

        with open(self.persistence_path, 'wb') as f:
            pickle.dump(state, f)
            os.fsync(f.fileno())

    def load_state(self) -> Optional[Tuple[List[Perspective], List[Iteration], int]]:
        if not os.path.exists(self.persistence_path):
            return None

        with open(self.persistence_path, 'rb') as f:
            state = pickle.load(f)

        return state["perspectives"], state["iterations"], state["iteration_count"]

    def should_fsync(self, iteration_count: int) -> bool:
        return iteration_count > 0 and iteration_count % self.fsync_interval == 0