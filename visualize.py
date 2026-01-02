#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

class AutopoeticVisualizer:
    def __init__(self, export_path="export"):
        self.export_path = Path(export_path)
        self.iterations_df = None
        self.adjustments_df = None
        self.load_data()

    def load_data(self):
        """Load CSV data from export directory"""
        iterations_file = self.export_path / "iterations.csv"
        adjustments_file = self.export_path / "semantic_adjustments.csv"

        if iterations_file.exists():
            self.iterations_df = pd.read_csv(iterations_file)
            print(f"Loaded {len(self.iterations_df)} iterations")
        else:
            print(f"Warning: {iterations_file} not found")

        if adjustments_file.exists():
            self.adjustments_df = pd.read_csv(adjustments_file)
            print(f"Loaded {len(self.adjustments_df)} semantic adjustments")
        else:
            print(f"Warning: {adjustments_file} not found")

    def plot_slow_ema_convergence(self, semantics_to_show=None, max_iterations=None):
        """Show slow EMA convergence for selected semantics"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        # Filter data
        df = self.adjustments_df.copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        # Get unique perspectives and semantics
        perspectives = df['perspective_name'].unique()
        all_semantics = df['semantic_name'].unique()

        # Select semantics to show (default to all semantics)
        if semantics_to_show is None:
            semantics_to_show = sorted(all_semantics)  # Show all semantics

        # Create dynamic grid layout for all dimensions
        n_dims = len(semantics_to_show)
        n_cols = 3
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        colors = plt.cm.Set1(np.linspace(0, 1, len(perspectives)))

        for i, semantic in enumerate(semantics_to_show):
            ax = axes[i]

            for j, perspective in enumerate(perspectives):
                # Filter data for this perspective and semantic
                mask = (df['perspective_name'] == perspective) & (df['semantic_name'] == semantic)
                data = df[mask].sort_values('iteration_id')

                if len(data) > 0:
                    # Convert slow_ema to numeric, handling empty strings
                    slow_ema = pd.to_numeric(data['slow_ema'], errors='coerce')

                    ax.plot(data['iteration_id'], slow_ema,
                           label=perspective, color=colors[j], alpha=0.8)

            ax.set_title(f'EMA Evolution: {semantic}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('EMA Value')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for i in range(len(semantics_to_show), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Autopoietic Learning: EMA Convergence Across Perspectives (all {n_dims} dimensions)',
                     fontsize=16, y=1.00)
        plt.savefig('ema_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_perspective_evolution(self, perspective_name, max_iterations=None):
        """Plot semantic evolution for a single perspective"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        # Filter data for this perspective
        df = self.adjustments_df[self.adjustments_df['perspective_name'] == perspective_name].copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        if len(df) == 0:
            print(f"No data found for perspective: {perspective_name}")
            return

        semantics = sorted(df['semantic_name'].unique())

        plt.figure(figsize=(15, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, len(semantics)))

        for i, semantic in enumerate(semantics):
            semantic_data = df[df['semantic_name'] == semantic].sort_values('iteration_id')

            if len(semantic_data) > 0:
                # Convert slow_ema to numeric
                slow_ema = pd.to_numeric(semantic_data['slow_ema'], errors='coerce')

                plt.plot(semantic_data['iteration_id'], slow_ema,
                        label=semantic, color=colors[i], alpha=0.8, linewidth=1.5)

        plt.title(f'Semantic Profile Evolution: {perspective_name}', fontsize=16)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('EMA Adjustment', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f'evolution_{perspective_name.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_perspective_evolutions(self, max_iterations=None):
        """Generate evolution plots for all perspectives"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        perspectives = self.adjustments_df['perspective_name'].unique()
        for perspective in perspectives:
            print(f"Plotting evolution for: {perspective}")
            self.plot_perspective_evolution(perspective, max_iterations)

    def plot_radar_chart_comparison(self, iteration_window=None):
        """Create radar/spider chart comparing final EMA states across perspectives"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        df = self.adjustments_df.copy()

        # Use latest data or specific iteration window
        if iteration_window:
            start_iter, end_iter = iteration_window
            df = df[(df['iteration_id'] >= start_iter) & (df['iteration_id'] <= end_iter)]

        # Get the latest EMA value for each perspective-semantic pair
        latest_data = df.groupby(['perspective_name', 'semantic_name'])['slow_ema'].last().reset_index()
        latest_data['slow_ema'] = pd.to_numeric(latest_data['slow_ema'], errors='coerce')

        # Pivot to get perspectives as columns, semantics as rows
        radar_data = latest_data.pivot(index='semantic_name', columns='perspective_name', values='slow_ema')
        radar_data = radar_data.fillna(0)  # Fill NaN with 0

        semantics = radar_data.index.tolist()
        perspectives = radar_data.columns.tolist()

        # Set up radar chart
        angles = [n / float(len(semantics)) * 2 * pi for n in range(len(semantics))]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set1(np.linspace(0, 1, len(perspectives)))

        for i, perspective in enumerate(perspectives):
            values = radar_data[perspective].tolist()
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=perspective, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(semantics, fontsize=10)
        ax.set_ylim(-1, 1)  # Adjust based on your data range
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        plt.title('Perspective Archetypes: Final Semantic Profiles',
                 fontsize=16, pad=20)

        plt.tight_layout()
        plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_semantic_drift_analysis(self, max_iterations=None):
        """Analyze how each semantic dimension evolves across all perspectives"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        df = self.adjustments_df.copy()
        if max_iterations:
            df = df[df['iteration_id'] <= max_iterations]

        semantics = sorted(df['semantic_name'].unique())
        perspectives = sorted(df['perspective_name'].unique())

        # Create a large figure for all semantics
        cols = 4
        rows = (len(semantics) + cols - 1) // cols  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []

        colors = plt.cm.Set1(np.linspace(0, 1, len(perspectives)))

        for i, semantic in enumerate(semantics):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get final EMA values for this semantic across all perspectives
            semantic_data = df[df['semantic_name'] == semantic]

            final_values = []
            perspective_labels = []

            for perspective in perspectives:
                persp_data = semantic_data[semantic_data['perspective_name'] == perspective]
                if len(persp_data) > 0:
                    latest_ema = pd.to_numeric(persp_data['slow_ema'].iloc[-1], errors='coerce')
                    if not pd.isna(latest_ema):
                        final_values.append(latest_ema)
                        perspective_labels.append(perspective)

            if final_values:
                # Create bar chart showing final EMA values
                bars = ax.bar(range(len(final_values)), final_values,
                             color=[colors[perspectives.index(p)] for p in perspective_labels])

                ax.set_title(f'{semantic}', fontsize=12)
                ax.set_ylabel('Final EMA', fontsize=10)
                ax.set_xticks(range(len(perspective_labels)))
                ax.set_xticklabels([p.split()[-1] for p in perspective_labels],
                                  rotation=45, fontsize=8)  # Use last word of perspective name
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, final_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # Hide unused subplots
        for i in range(len(semantics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Semantic Drift Analysis: Final EMA Values Across Perspectives (all {len(semantics)} dimensions)',
                     fontsize=16, y=1.00)
        plt.savefig('semantic_drift_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """Generate a text summary of the autopoietic behavior"""
        if self.adjustments_df is None:
            print("No adjustment data available for summary")
            return

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        print("\n" + "="*60)
        print("AUTOPOIETIC SYSTEM ANALYSIS SUMMARY")
        print("="*60)

        # Get final EMA values for each perspective
        perspectives = sorted(df['perspective_name'].unique())
        semantics = sorted(df['semantic_name'].unique())

        print(f"\nSystem Configuration:")
        print(f"  Perspectives: {len(perspectives)}")
        print(f"  Semantic Dimensions: {len(semantics)}")
        print(f"  Total Iterations Analyzed: {df['iteration_id'].max()}")

        print(f"\nEmergent Personality Archetypes:")
        for perspective in perspectives:
            persp_data = df[df['perspective_name'] == perspective]

            # Find semantics where this perspective has strongest/weakest adaptation
            final_emas = {}
            for semantic in semantics:
                semantic_data = persp_data[persp_data['semantic_name'] == semantic]
                if len(semantic_data) > 0:
                    final_ema = semantic_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        final_emas[semantic] = final_ema

            if final_emas:
                # Find extremes
                max_semantic = max(final_emas, key=final_emas.get)
                min_semantic = min(final_emas, key=final_emas.get)

                print(f"\n  {perspective}:")
                print(f"    Strongest adaptation: {max_semantic} ({final_emas[max_semantic]:.3f})")
                print(f"    Weakest adaptation: {min_semantic} ({final_emas[min_semantic]:.3f})")
                print(f"    Adaptation range: {final_emas[max_semantic] - final_emas[min_semantic]:.3f}")

        print(f"\nSemantic Dimension Analysis:")
        # Analyze which semantics show most/least variation across perspectives
        semantic_variations = {}
        for semantic in semantics:
            semantic_values = []
            for perspective in perspectives:
                persp_semantic = df[(df['perspective_name'] == perspective) &
                                  (df['semantic_name'] == semantic)]
                if len(persp_semantic) > 0:
                    final_ema = persp_semantic['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        semantic_values.append(final_ema)

            if len(semantic_values) > 1:
                variation = max(semantic_values) - min(semantic_values)
                semantic_variations[semantic] = variation

        if semantic_variations:
            most_varied = max(semantic_variations, key=semantic_variations.get)
            least_varied = min(semantic_variations, key=semantic_variations.get)

            print(f"  Most differentiated: {most_varied} (range: {semantic_variations[most_varied]:.3f})")
            print(f"  Least differentiated: {least_varied} (range: {semantic_variations[least_varied]:.3f})")

        print(f"\nConclusion:")
        print(f"  The system demonstrates emergent personality archetypes through")
        print(f"  differential adaptation patterns across semantic dimensions.")
        print("="*60)

    def plot_competency_vs_emergence(self):
        """Compare relative positioning: initial competency ranks vs final EMA ranks"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        # Import perspectives to get initial competencies
        from perspectives import Perspectives

        # Get final EMA values
        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        semantics = sorted(df['semantic_name'].unique())

        # Create comparison plots for each semantic
        cols = 4
        rows = (len(semantics) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []

        for i, semantic in enumerate(semantics):
            if i >= len(axes):
                break

            ax = axes[i]

            # Collect data for ranking
            perspective_data = []

            for perspective_obj in Perspectives:
                perspective_name = perspective_obj.name

                # Get initial competency
                initial_comp = float(perspective_obj.semantic_profile.semantics.get(semantic, 0))

                # Get final EMA
                persp_data = df[(df['perspective_name'] == perspective_name) &
                               (df['semantic_name'] == semantic)]
                if len(persp_data) > 0:
                    final_ema = persp_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        # Calculate effective final competency (initial + adaptation)
                        combined_final = initial_comp + float(final_ema)
                        perspective_data.append((perspective_name, initial_comp, combined_final))

            if len(perspective_data) >= 2:
                # Sort by initial competency to get initial ranks (1 = highest)
                initial_sorted = sorted(perspective_data, key=lambda x: x[1], reverse=True)
                initial_ranks = {name: rank+1 for rank, (name, _, _) in enumerate(initial_sorted)}

                # Sort by final effective competency to get final ranks (1 = highest)
                final_sorted = sorted(perspective_data, key=lambda x: x[2], reverse=True)
                final_ranks = {name: rank+1 for rank, (name, _, _) in enumerate(final_sorted)}

                # Create scatter plot of ranks
                initial_rank_list = []
                final_rank_list = []
                names = []

                for name, _, _ in perspective_data:
                    initial_rank_list.append(initial_ranks[name])
                    final_rank_list.append(final_ranks[name])
                    names.append(name)

                colors = plt.cm.Set1(np.linspace(0, 1, len(names)))
                scatter = ax.scatter(initial_rank_list, final_rank_list,
                                   c=colors, s=100, alpha=0.7)

                # Add diagonal line showing "no rank change"
                max_rank = max(max(initial_rank_list), max(final_rank_list))
                ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.3, label='No Change')

                # Label points with perspective abbreviations
                for x, y, name in zip(initial_rank_list, final_rank_list, names):
                    abbrev = ''.join([word[0] for word in name.split()])
                    ax.annotate(abbrev, (x, y), xytext=(5, 5),
                              textcoords='offset points', fontsize=8)

                    # Highlight rank changes
                    if abs(y - x) >= 1:  # Significant rank change
                        if y < x:  # Better final rank (lower number)
                            ax.annotate('↑', (x, y), xytext=(0, 15),
                                      textcoords='offset points', fontsize=12,
                                      color='green', ha='center')
                        else:  # Worse final rank (higher number)
                            ax.annotate('↓', (x, y), xytext=(0, -20),
                                      textcoords='offset points', fontsize=12,
                                      color='red', ha='center')

                ax.set_title(f'{semantic}', fontsize=12)
                ax.set_xlabel('Initial Competency Rank', fontsize=10)
                ax.set_ylabel('Final Effective Competency Rank', fontsize=10)
                ax.invert_yaxis()  # Lower rank numbers at top
                ax.invert_xaxis()  # Lower rank numbers at left
                ax.grid(True, alpha=0.3)

                # Set integer ticks
                ax.set_xticks(range(1, max_rank + 1))
                ax.set_yticks(range(1, max_rank + 1))

        # Hide unused subplots
        for i in range(len(semantics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Rank Changes: Initial vs Final Effective Competency (all {len(semantics)} dimensions)\n(↑ = Improved Rank, ↓ = Declined Rank)',
                     fontsize=16, y=1.00)
        plt.savefig('competency_vs_emergence_ranks.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_unexpected_outcomes(self):
        """Generate detailed analysis of rank changes and surprising emergent behaviors"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        from perspectives import Perspectives

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        print("\n" + "="*70)
        print("RANK CHANGE ANALYSIS: Who Ended Up Where vs. Initial Design")
        print("="*70)

        semantics = sorted(df['semantic_name'].unique())
        all_rank_changes = []

        for semantic in semantics:
            # Collect data for this semantic
            perspective_data = []

            for perspective_obj in Perspectives:
                perspective_name = perspective_obj.name
                initial_comp = float(perspective_obj.semantic_profile.semantics.get(semantic, 0))

                persp_data = df[(df['perspective_name'] == perspective_name) &
                               (df['semantic_name'] == semantic)]
                if len(persp_data) > 0:
                    final_ema = persp_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        # Calculate effective final competency (initial + adaptation)
                        combined_final = initial_comp + float(final_ema)
                        perspective_data.append((perspective_name, initial_comp, combined_final))

            if len(perspective_data) >= 2:
                # Calculate initial and final effective competency ranks
                initial_sorted = sorted(perspective_data, key=lambda x: x[1], reverse=True)
                initial_ranks = {name: rank+1 for rank, (name, _, _) in enumerate(initial_sorted)}

                final_sorted = sorted(perspective_data, key=lambda x: x[2], reverse=True)
                final_ranks = {name: rank+1 for rank, (name, _, _) in enumerate(final_sorted)}

                # Find rank changes
                rank_changes = []
                for name, initial_comp, combined_final in perspective_data:
                    initial_rank = initial_ranks[name]
                    final_rank = final_ranks[name]
                    rank_change = initial_rank - final_rank  # Positive = improved rank

                    if abs(rank_change) >= 1:  # Significant rank change
                        rank_changes.append((name, semantic, initial_rank, final_rank, rank_change))
                        all_rank_changes.append((name, semantic, initial_rank, final_rank, rank_change, abs(rank_change)))

                # Show this semantic's surprises
                if rank_changes:
                    print(f"\n{semantic}:")
                    print(f"  Initial order: {' → '.join([name.split()[-1] for name, _, _ in initial_sorted])}")
                    print(f"  Final order:   {' → '.join([name.split()[-1] for name, _, _ in final_sorted])}")

                    for name, _, initial_rank, final_rank, rank_change in sorted(rank_changes, key=lambda x: abs(x[4]), reverse=True):
                        direction = "↗" if rank_change > 0 else "↘"
                        action = "rose" if rank_change > 0 else "fell"
                        print(f"    {name.split()[-1]}: #{initial_rank} → #{final_rank} ({action} {abs(rank_change)} positions) {direction}")

        # Summary of biggest rank changes
        print(f"\nTOP 10 BIGGEST RANK CHANGES:")
        all_rank_changes.sort(key=lambda x: x[5], reverse=True)  # Sort by magnitude

        for i, (name, semantic, initial_rank, final_rank, rank_change, magnitude) in enumerate(all_rank_changes[:10]):
            direction = "↗" if rank_change > 0 else "↘"
            action = "rose" if rank_change > 0 else "fell"
            print(f"{i+1:2d}. {name} / {semantic}: #{initial_rank} → #{final_rank} ({action} {magnitude} positions) {direction}")

        print(f"\nKey Insights:")

        # Count rank improvements vs declines
        improvements = sum(1 for _, _, _, _, change, _ in all_rank_changes if change > 0)
        declines = sum(1 for _, _, _, _, change, _ in all_rank_changes if change < 0)

        print(f"  • {improvements} significant rank improvements vs {declines} declines")
        print(f"  • Consensus-driven adaptation reshuffled {len(all_rank_changes)} perspective-semantic rankings")
        print(f"  • Initial competency design ≠ final emergent specialization")
        print("  • The system developed its own hierarchy through collective dynamics")
        print("="*70)

def main():
    """Generate all visualizations"""
    viz = AutopoeticVisualizer()

    print("Generating EMA convergence visualization...")
    viz.plot_slow_ema_convergence()

    print("\nGenerating individual perspective evolutions...")
    viz.plot_all_perspective_evolutions()

    print("\nGenerating radar chart comparison...")
    viz.plot_radar_chart_comparison()

    print("\nGenerating semantic drift analysis...")
    viz.plot_semantic_drift_analysis()

    print("\nGenerating summary report...")
    viz.generate_summary_report()

    print("\nAnalyzing designed vs emergent behaviors...")
    viz.plot_competency_vs_emergence()

    print("\nAnalyzing unexpected outcomes...")
    viz.analyze_unexpected_outcomes()

    print("\nVisualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()
