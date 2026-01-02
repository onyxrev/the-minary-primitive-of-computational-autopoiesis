#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from engine import SEMANTICS

class ConsensusVisualizer:
    def __init__(self, export_path="export"):
        self.export_path = Path(export_path)
        self.iterations_df = None
        self.load_data()

    def load_data(self):
        """Load iterations CSV data"""
        iterations_file = self.export_path / "iterations.csv"

        if iterations_file.exists():
            self.iterations_df = pd.read_csv(iterations_file)
            print(f"Loaded {len(self.iterations_df)} iterations")
            print(f"Columns: {list(self.iterations_df.columns)}")
        else:
            print(f"Warning: {iterations_file} not found")

    def get_consensus_columns(self):
        """Get consensus column names"""
        if self.iterations_df is None:
            return []
        return [col for col in self.iterations_df.columns if col.startswith('consensus_') and not col.startswith('normalized_consensus_')]

    def get_normalized_consensus_columns(self):
        """Get normalized consensus column names"""
        if self.iterations_df is None:
            return []
        return [col for col in self.iterations_df.columns if col.startswith('normalized_consensus_')]

    def get_semantic_name(self, column_name):
        """Map column name to semantic dimension name"""
        # Extract index from column name (e.g., 'consensus_5' -> 5)
        if '_' in column_name:
            try:
                index = int(column_name.split('_')[-1])
                if 0 <= index < len(SEMANTICS):
                    return SEMANTICS[index]
            except (ValueError, IndexError):
                pass
        return column_name

    def plot_raw_consensus_evolution(self, max_iterations=None, dimensions_to_show=None):
        """Plot raw consensus values over time to see underlying dynamics"""
        if self.iterations_df is None:
            print("No iteration data available")
            return

        df = self.iterations_df.copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        consensus_cols = self.get_consensus_columns()

        if not consensus_cols:
            print("No consensus columns found")
            return

        # Show all dimensions by default
        if dimensions_to_show is None:
            selected_cols = consensus_cols
        else:
            selected_cols = consensus_cols[:dimensions_to_show]

        # Convert consensus values to numeric
        for col in selected_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create grid layout - 19 dimensions in rows of 3
        n_dims = len(selected_cols)
        n_cols = 3
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(selected_cols):
            ax = axes[i]
            semantic_name = self.get_semantic_name(col)

            # Plot consensus evolution
            ax.plot(df['iteration_id'], df[col], alpha=0.7, linewidth=0.5)

            # Add running average to see trends
            window = max(1, len(df) // 100)  # 1% of data as window
            if window > 1:
                running_avg = df[col].rolling(window=window, center=True).mean()
                ax.plot(df['iteration_id'], running_avg, 'r-', linewidth=2, label=f'Running avg ({window})')
                ax.legend()

            ax.set_title(f'{semantic_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Consensus Value')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(selected_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Raw Consensus Dynamics (all {len(selected_cols)} dimensions)', fontsize=16, y=1.00)
        plt.savefig('raw_consensus_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_consensus_statistics(self, max_iterations=None):
        """Generate statistical summary of consensus behavior"""
        if self.iterations_df is None:
            print("No iteration data available")
            return

        df = self.iterations_df.copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        consensus_cols = self.get_consensus_columns()
        normalized_cols = self.get_normalized_consensus_columns()

        print("\n" + "="*60)
        print("CONSENSUS DYNAMICS STATISTICAL ANALYSIS")
        print("="*60)

        print(f"\nDataset: {len(df)} iterations analyzed")
        print(f"Total dimensions: {len(consensus_cols)}")

        # Raw consensus statistics
        print(f"\nRAW CONSENSUS STATISTICS:")
        for col in consensus_cols:
            semantic_name = self.get_semantic_name(col)
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 0:
                print(f"  {semantic_name}:")
                print(f"    Range: [{values.min():.3f}, {values.max():.3f}]")
                print(f"    Mean: {values.mean():.3f}, Std: {values.std():.3f}")
                print(f"    Zero count: {(values == 0).sum()} ({(values == 0).mean()*100:.1f}%)")
            else:
                print(f"  {semantic_name}: No data")

        # Normalized consensus statistics
        print(f"\nNORMALIZED CONSENSUS STATISTICS:")
        for col in normalized_cols:
            semantic_name = self.get_semantic_name(col)
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 0:
                print(f"  {semantic_name}:")
                print(f"    Range: [{values.min():.3f}, {values.max():.3f}]")
                print(f"    Mean: {values.mean():.3f}, Std: {values.std():.3f}")
            else:
                print(f"  {semantic_name}: No data")

        # Analyze early vs late behavior for dimensions with data
        if len(df) > 1000:
            early_df = df.head(1000)
            late_df = df.tail(1000)

            print(f"\nEARLY vs LATE BEHAVIOR COMPARISON:")
            print(f"  (First 1000 vs Last 1000 iterations)")

            # Show first 3 dimensions that have data
            shown = 0
            for col in consensus_cols:
                if shown >= 3:
                    break
                semantic_name = self.get_semantic_name(col)
                early_vals = pd.to_numeric(early_df[col], errors='coerce').dropna()
                late_vals = pd.to_numeric(late_df[col], errors='coerce').dropna()

                if len(early_vals) > 0 and len(late_vals) > 0:
                    print(f"  {semantic_name}:")
                    print(f"    Early std: {early_vals.std():.3f}, Late std: {late_vals.std():.3f}")
                    print(f"    Early zeros: {(early_vals == 0).mean()*100:.1f}%, Late zeros: {(late_vals == 0).mean()*100:.1f}%")
                    shown += 1

        print("="*60)

    def plot_consensus_distribution(self, max_iterations=None, dimensions_to_show=None):
        """Plot histograms of consensus value distributions"""
        if self.iterations_df is None:
            print("No iteration data available")
            return

        df = self.iterations_df.copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        consensus_cols = self.get_consensus_columns()

        if not consensus_cols:
            print("No consensus columns found")
            return

        # Show all dimensions by default
        if dimensions_to_show is None:
            selected_cols = consensus_cols
        else:
            selected_cols = consensus_cols[:dimensions_to_show]

        n_dims = len(selected_cols)
        n_cols = 3
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(selected_cols):
            semantic_name = self.get_semantic_name(col)
            values = pd.to_numeric(df[col], errors='coerce').dropna()

            ax = axes[i]

            if len(values) > 0:
                # Create histogram
                ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{semantic_name}')
                ax.set_xlabel('Consensus Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

                # Add statistics text
                stats_text = f'Mean: {values.mean():.3f}\nStd: {values.std():.3f}\nZeros: {(values == 0).sum()}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No data for this dimension
                ax.set_title(f'{semantic_name}')
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16, color='gray')
                ax.set_xlabel('Consensus Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Consensus Value Distributions (all {n_dims} dimensions)', fontsize=16, y=1.00)
        plt.savefig('consensus_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Generate consensus analysis visualizations"""
    viz = ConsensusVisualizer()

    print("Analyzing raw consensus dynamics...")
    viz.analyze_consensus_statistics()

    print("\nGenerating consensus evolution plots...")
    viz.plot_raw_consensus_evolution()

    print("\nGenerating consensus distribution analysis...")
    viz.plot_consensus_distribution()

    print("\nConsensus analysis complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()