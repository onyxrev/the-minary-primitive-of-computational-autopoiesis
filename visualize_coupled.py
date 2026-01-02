#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path
from itertools import combinations
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict

class CoupledVisualizationAnalyzer:
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

    def plot_alliance_formation_matrix(self, iteration_windows=None):
        """Visualize alliance formation through correlation heatmaps across time windows"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        from perspectives import Perspectives

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        max_iteration = df['iteration_id'].max()

        # Define time windows for alliance analysis
        if iteration_windows is None:
            window_size = max_iteration // 4
            iteration_windows = [
                (1, window_size),
                (window_size, 2 * window_size),
                (2 * window_size, 3 * window_size),
                (3 * window_size, max_iteration)
            ]

        perspectives = [p.name for p in Perspectives]
        semantics = sorted(df['semantic_name'].unique())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for window_idx, (start_iter, end_iter) in enumerate(iteration_windows):
            ax = axes[window_idx]

            # Filter data for this time window
            window_data = df[(df['iteration_id'] >= start_iter) & (df['iteration_id'] <= end_iter)]

            # Create correlation matrix between perspectives
            correlation_matrix = np.zeros((len(perspectives), len(perspectives)))

            for i, persp1 in enumerate(perspectives):
                for j, persp2 in enumerate(perspectives):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # Calculate correlation across all semantics for this time window
                        persp1_values = []
                        persp2_values = []

                        for semantic in semantics:
                            persp1_data = window_data[
                                (window_data['perspective_name'] == persp1) &
                                (window_data['semantic_name'] == semantic)
                            ]
                            persp2_data = window_data[
                                (window_data['perspective_name'] == persp2) &
                                (window_data['semantic_name'] == semantic)
                            ]

                            if len(persp1_data) > 0 and len(persp2_data) > 0:
                                persp1_avg = persp1_data['slow_ema'].mean()
                                persp2_avg = persp2_data['slow_ema'].mean()
                                if not (pd.isna(persp1_avg) or pd.isna(persp2_avg)):
                                    persp1_values.append(persp1_avg)
                                    persp2_values.append(persp2_avg)

                        if len(persp1_values) > 1:
                            corr, _ = pearsonr(persp1_values, persp2_values)
                            correlation_matrix[i, j] = corr if not np.isnan(corr) else 0

            # Create heatmap
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

            # Add colorbar for first subplot only
            if window_idx == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Alliance Strength (Correlation)', fontsize=10)

            # Customize appearance
            ax.set_xticks(range(len(perspectives)))
            ax.set_yticks(range(len(perspectives)))
            ax.set_xticklabels([p.split()[-1] for p in perspectives], rotation=45, fontsize=8)
            ax.set_yticklabels([p.split()[-1] for p in perspectives], fontsize=8)
            ax.set_title(f'Iterations {start_iter}-{end_iter}', fontsize=12)

            # Add correlation values as text
            for i in range(len(perspectives)):
                for j in range(len(perspectives)):
                    if i != j:  # Don't show 1.0 on diagonal
                        text_color = 'white' if abs(correlation_matrix[i, j]) > 0.5 else 'black'
                        ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha='center', va='center', color=text_color, fontsize=8)

        plt.tight_layout()
        plt.suptitle('Alliance Formation Over Time: Perspective Correlation Matrices\n(Red = Strong Alliance, Blue = Opposition)',
                     fontsize=14, y=0.98)
        plt.savefig('alliance_formation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_competency_overlap_clusters(self):
        """Visualize which perspectives form clusters based on initial competency overlap"""
        from perspectives import Perspectives

        perspective_names = [p.name for p in Perspectives]
        semantics = list(Perspectives[0].semantic_profile.semantics.keys())

        # Create competency matrix
        competency_matrix = np.zeros((len(perspective_names), len(semantics)))

        for i, perspective in enumerate(Perspectives):
            for j, semantic in enumerate(semantics):
                competency_matrix[i, j] = float(perspective.semantic_profile.semantics[semantic])

        # Calculate perspective similarity matrix
        similarity_matrix = np.corrcoef(competency_matrix)

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Initial competency heatmap
        im1 = ax1.imshow(competency_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Initial Competency Profiles', fontsize=14)
        ax1.set_xlabel('Semantic Dimensions', fontsize=12)
        ax1.set_ylabel('Perspectives', fontsize=12)
        ax1.set_xticks(range(len(semantics)))
        ax1.set_xticklabels(semantics, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks(range(len(perspective_names)))
        ax1.set_yticklabels([p.split()[-1] for p in perspective_names], fontsize=10)

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Competency Level', fontsize=10)

        # Plot 2: Similarity matrix (potential alliance prediction)
        im2 = ax2.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title('Competency Overlap Similarity\n(Predicts Potential Alliances)', fontsize=14)
        ax2.set_xlabel('Perspectives', fontsize=12)
        ax2.set_ylabel('Perspectives', fontsize=12)
        ax2.set_xticks(range(len(perspective_names)))
        ax2.set_xticklabels([p.split()[-1] for p in perspective_names], rotation=45, fontsize=10)
        ax2.set_yticks(range(len(perspective_names)))
        ax2.set_yticklabels([p.split()[-1] for p in perspective_names], fontsize=10)

        # Add similarity values as text
        for i in range(len(perspective_names)):
            for j in range(len(perspective_names)):
                if i != j:
                    text_color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
                    ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=9)

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Similarity Score', fontsize=10)

        plt.tight_layout()
        plt.savefig('competency_overlap_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print analysis
        print("\nCOMPETENCY OVERLAP ANALYSIS:")
        print("="*50)

        # Find strongest potential alliances
        strongest_pairs = []
        for i in range(len(perspective_names)):
            for j in range(i+1, len(perspective_names)):
                similarity = similarity_matrix[i, j]
                strongest_pairs.append((perspective_names[i], perspective_names[j], similarity))

        strongest_pairs.sort(key=lambda x: x[2], reverse=True)

        print("Strongest Predicted Alliances (based on competency overlap):")
        for i, (persp1, persp2, similarity) in enumerate(strongest_pairs[:5]):
            print(f"{i+1}. {persp1.split()[-1]} ↔ {persp2.split()[-1]}: {similarity:.3f}")

        print("\nWeakest Potential Alliances (most likely to oppose):")
        for i, (persp1, persp2, similarity) in enumerate(strongest_pairs[-5:]):
            print(f"{i+1}. {persp1.split()[-1]} ↔ {persp2.split()[-1]}: {similarity:.3f}")

    def plot_halo_effect_analysis(self, window_size=150):
        """Analyze dynamic halo effects - when high competency dimensions influence others over time"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        from perspectives import Perspectives

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        perspectives = [p.name for p in Perspectives]
        semantics = sorted(df['semantic_name'].unique())

        # For each perspective, identify their highest initial competency dimensions
        competency_leaders = {}
        for perspective_obj in Perspectives:
            perspective_name = perspective_obj.name
            competencies = perspective_obj.semantic_profile.semantics

            # Find their top 3 competencies
            top_competencies = sorted(competencies.items(), key=lambda x: x[1], reverse=True)[:3]
            competency_leaders[perspective_name] = [semantic for semantic, _ in top_competencies]

        fig, axes = plt.subplots(len(perspectives), 1, figsize=(15, 4 * len(perspectives)))
        if len(perspectives) == 1:
            axes = [axes]

        for persp_idx, perspective in enumerate(perspectives):
            ax = axes[persp_idx]

            # Get this perspective's data
            persp_data = df[df['perspective_name'] == perspective].copy()

            # Separate leader vs non-leader semantics
            leader_semantics = competency_leaders[perspective]
            non_leader_semantics = [s for s in semantics if s not in leader_semantics]

            # Calculate dynamic halo differential over time
            iterations = sorted(persp_data['iteration_id'].unique())

            halo_differential = []
            rolling_correlation = []

            for iteration in iterations:
                iter_data = persp_data[persp_data['iteration_id'] == iteration]

                # Average EMA for leader semantics
                leader_emas = []
                for semantic in leader_semantics:
                    semantic_data = iter_data[iter_data['semantic_name'] == semantic]
                    if len(semantic_data) > 0:
                        ema_val = semantic_data['slow_ema'].iloc[0]
                        if not pd.isna(ema_val):
                            leader_emas.append(float(ema_val))

                # Average EMA for non-leader semantics
                non_leader_emas = []
                for semantic in non_leader_semantics:
                    semantic_data = iter_data[iter_data['semantic_name'] == semantic]
                    if len(semantic_data) > 0:
                        ema_val = semantic_data['slow_ema'].iloc[0]
                        if not pd.isna(ema_val):
                            non_leader_emas.append(float(ema_val))

                # Calculate the halo differential (leader avg - non-leader avg)
                leader_avg = np.mean(leader_emas) if leader_emas else 0
                non_leader_avg = np.mean(non_leader_emas) if non_leader_emas else 0

                halo_differential.append(leader_avg - non_leader_avg)

            # Apply lighter smoothing to the differential
            if len(iterations) > window_size:
                halo_smooth = pd.Series(halo_differential).rolling(window=window_size, min_periods=1).mean()
            else:
                halo_smooth = pd.Series(halo_differential)

            # Calculate rolling correlation between high-competency and other dimensions
            correlation_window = min(window_size * 2, len(iterations) // 4)
            if correlation_window > 10:
                leader_trajectory = []
                non_leader_trajectory = []

                for iteration in iterations:
                    iter_data = persp_data[persp_data['iteration_id'] == iteration]

                    leader_emas = [float(iter_data[iter_data['semantic_name'] == s]['slow_ema'].iloc[0])
                                 for s in leader_semantics
                                 if len(iter_data[iter_data['semantic_name'] == s]) > 0
                                 and not pd.isna(iter_data[iter_data['semantic_name'] == s]['slow_ema'].iloc[0])]

                    non_leader_emas = [float(iter_data[iter_data['semantic_name'] == s]['slow_ema'].iloc[0])
                                     for s in non_leader_semantics
                                     if len(iter_data[iter_data['semantic_name'] == s]) > 0
                                     and not pd.isna(iter_data[iter_data['semantic_name'] == s]['slow_ema'].iloc[0])]

                    leader_trajectory.append(np.mean(leader_emas) if leader_emas else 0)
                    non_leader_trajectory.append(np.mean(non_leader_emas) if non_leader_emas else 0)

                # Calculate rolling correlation
                rolling_corr_values = []
                for i in range(len(iterations)):
                    start_idx = max(0, i - correlation_window // 2)
                    end_idx = min(len(iterations), i + correlation_window // 2)

                    if end_idx - start_idx > 10:
                        window_leader = leader_trajectory[start_idx:end_idx]
                        window_non_leader = non_leader_trajectory[start_idx:end_idx]

                        if len(window_leader) > 1 and np.std(window_leader) > 0 and np.std(window_non_leader) > 0:
                            corr, _ = pearsonr(window_leader, window_non_leader)
                            rolling_corr_values.append(corr if not np.isnan(corr) else 0)
                        else:
                            rolling_corr_values.append(0)
                    else:
                        rolling_corr_values.append(0)

                # Plot rolling correlation as background
                ax2 = ax.twinx()
                ax2.fill_between(iterations, rolling_corr_values, alpha=0.3, color='purple',
                               label='Synchronization Strength')
                ax2.set_ylabel('Correlation\n(Sync Strength)', fontsize=10, color='purple')
                ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
                ax2.set_ylim(-1, 1)

            # Plot the main halo differential
            ax.plot(iterations, halo_smooth, 'g-', linewidth=3, alpha=0.8,
                   label=f'Halo Strength (High - Other)')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

            # Add statistics
            final_halo = halo_smooth.iloc[-1] if len(halo_smooth) > 0 else 0
            max_halo = halo_smooth.max() if len(halo_smooth) > 0 else 0
            min_halo = halo_smooth.min() if len(halo_smooth) > 0 else 0

            ax.text(0.02, 0.98, f'Final Halo: {final_halo:.3f}\nRange: {min_halo:.3f} to {max_halo:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
                   verticalalignment='top', fontsize=10)

            ax.set_title(f'{perspective}: Dynamic Halo Effect Over Time', fontsize=12)
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Halo Differential\n(High Comp - Others)', fontsize=10)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Highlight high competency semantics
            ax.text(0.98, 0.02, f'High competencies: {", ".join(leader_semantics)}',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
                   horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        plt.tight_layout()
        plt.suptitle('Dynamic Halo Effect: How High Competencies Influence Others Over Time\n(Positive = Strong Halo, Purple = Synchronization)',
                     fontsize=14, y=0.99)
        plt.savefig('halo_effect_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_synchronization_dynamics(self, window_size=500):
        """Visualize how perspectives synchronize their responses over time"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        from perspectives import Perspectives

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        perspectives = [p.name for p in Perspectives]
        semantics = sorted(df['semantic_name'].unique())  # Show all dimensions

        # Create dynamic grid layout for all dimensions
        n_dims = len(semantics)
        n_cols = 3
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for sem_idx, semantic in enumerate(semantics):
            ax = axes[sem_idx]

            # Get data for this semantic across all perspectives
            semantic_data = df[df['semantic_name'] == semantic].copy()

            # Create time series for each perspective
            perspective_trajectories = {}
            all_iterations = sorted(semantic_data['iteration_id'].unique())

            for perspective in perspectives:
                persp_data = semantic_data[semantic_data['perspective_name'] == perspective]

                trajectory = []
                for iteration in all_iterations:
                    iter_data = persp_data[persp_data['iteration_id'] == iteration]
                    if len(iter_data) > 0:
                        ema_val = iter_data['slow_ema'].iloc[0]
                        trajectory.append(float(ema_val) if not pd.isna(ema_val) else None)
                    else:
                        trajectory.append(None)

                # Apply smoothing
                trajectory_series = pd.Series(trajectory)
                if len(all_iterations) > window_size:
                    smoothed = trajectory_series.rolling(window=window_size, min_periods=1).mean()
                else:
                    smoothed = trajectory_series.ffill()

                perspective_trajectories[perspective] = smoothed

            # Plot trajectories
            colors = plt.cm.Set1(np.linspace(0, 1, len(perspectives)))
            for i, (perspective, trajectory) in enumerate(perspective_trajectories.items()):
                valid_data = trajectory.dropna()
                if len(valid_data) > 0:
                    ax.plot(all_iterations[:len(trajectory)], trajectory,
                           label=perspective.split()[-1], color=colors[i], alpha=0.8, linewidth=2)

            # Calculate and display synchronization metrics
            # Standard deviation across perspectives at each time point
            sync_variance = []
            for i in range(len(all_iterations)):
                values_at_time = []
                for trajectory in perspective_trajectories.values():
                    if i < len(trajectory) and not pd.isna(trajectory.iloc[i]):
                        values_at_time.append(trajectory.iloc[i])

                if len(values_at_time) > 1:
                    sync_variance.append(np.std(values_at_time))
                else:
                    sync_variance.append(np.nan)

            # Add synchronization trend as background
            if len(sync_variance) > 0:
                # Invert synchronization variance (lower variance = higher sync)
                max_var = np.nanmax(sync_variance)
                if max_var > 0:
                    sync_strength = [(max_var - v) / max_var if not np.isnan(v) else 0 for v in sync_variance]
                    ax2 = ax.twinx()
                    ax2.fill_between(all_iterations[:len(sync_strength)], sync_strength,
                                    alpha=0.2, color='gold', label='Synchronization')
                    ax2.set_ylabel('Synchronization\n(High = Similar)', fontsize=8, color='gold')
                    ax2.tick_params(axis='y', labelcolor='gold', labelsize=8)
                    ax2.set_ylim(0, 1)

            ax.set_title(f'{semantic}', fontsize=12)
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('EMA Value', fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'Synchronization Dynamics: How Perspectives Align Over Time (all {n_dims} dimensions)\n(Gold background = High synchronization)',
                     fontsize=14, y=1.00)
        plt.savefig('synchronization_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_coupled_emergence_patterns(self):
        """Generate comprehensive analysis of coupled strategy patterns"""
        if self.adjustments_df is None:
            print("No adjustment data available")
            return

        from perspectives import Perspectives

        df = self.adjustments_df.copy()
        df['slow_ema'] = pd.to_numeric(df['slow_ema'], errors='coerce')

        print("\n" + "="*70)
        print("COUPLED STRATEGY EMERGENCE ANALYSIS")
        print("="*70)

        perspectives = [p.name for p in Perspectives]
        semantics = sorted(df['semantic_name'].unique())

        # 1. Alliance strength analysis
        print("\n1. FINAL ALLIANCE STRENGTH (Cross-semantic correlation):")
        print("-" * 50)

        # Calculate final correlations between all perspective pairs
        alliance_strengths = []
        for persp1, persp2 in combinations(perspectives, 2):
            persp1_values = []
            persp2_values = []

            for semantic in semantics:
                persp1_data = df[(df['perspective_name'] == persp1) & (df['semantic_name'] == semantic)]
                persp2_data = df[(df['perspective_name'] == persp2) & (df['semantic_name'] == semantic)]

                if len(persp1_data) > 0 and len(persp2_data) > 0:
                    persp1_final = persp1_data['slow_ema'].iloc[-1]
                    persp2_final = persp2_data['slow_ema'].iloc[-1]

                    if not (pd.isna(persp1_final) or pd.isna(persp2_final)):
                        persp1_values.append(float(persp1_final))
                        persp2_values.append(float(persp2_final))

            if len(persp1_values) > 1:
                corr, p_value = pearsonr(persp1_values, persp2_values)
                alliance_strengths.append((persp1, persp2, corr, p_value))

        # Sort by alliance strength
        alliance_strengths.sort(key=lambda x: x[2], reverse=True)

        print("Strongest Alliances:")
        for i, (persp1, persp2, corr, p_val) in enumerate(alliance_strengths[:5]):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{i+1}. {persp1.split()[-1]} ↔ {persp2.split()[-1]}: {corr:.3f} {significance}")

        print("\nStrongest Oppositions:")
        for i, (persp1, persp2, corr, p_val) in enumerate(alliance_strengths[-5:]):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{i+1}. {persp1.split()[-1]} ↔ {persp2.split()[-1]}: {corr:.3f} {significance}")

        # 2. Halo effect analysis
        print(f"\n2. HALO EFFECT SUMMARY:")
        print("-" * 50)

        for perspective_obj in Perspectives:
            perspective_name = perspective_obj.name
            competencies = perspective_obj.semantic_profile.semantics

            # Find top competencies
            top_competencies = sorted(competencies.items(), key=lambda x: x[1], reverse=True)[:3]
            other_competencies = sorted(competencies.items(), key=lambda x: x[1], reverse=True)[3:]

            top_semantics = [semantic for semantic, _ in top_competencies]
            other_semantics = [semantic for semantic, _ in other_competencies]

            # Get final EMA values for both groups
            persp_data = df[df['perspective_name'] == perspective_name]

            top_final_emas = []
            other_final_emas = []

            for semantic in top_semantics:
                semantic_data = persp_data[persp_data['semantic_name'] == semantic]
                if len(semantic_data) > 0:
                    final_ema = semantic_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        top_final_emas.append(float(final_ema))

            for semantic in other_semantics:
                semantic_data = persp_data[persp_data['semantic_name'] == semantic]
                if len(semantic_data) > 0:
                    final_ema = semantic_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        other_final_emas.append(float(final_ema))

            if len(top_final_emas) > 0 and len(other_final_emas) > 0:
                top_avg = np.mean(top_final_emas)
                other_avg = np.mean(other_final_emas)
                halo_effect = top_avg - other_avg

                print(f"{perspective_name.split()[-1]:12}: Halo = {halo_effect:+.3f} "
                      f"(Top: {top_avg:.3f}, Others: {other_avg:.3f})")

        # 3. Synchronization analysis
        print(f"\n3. SYNCHRONIZATION CONVERGENCE:")
        print("-" * 50)

        # Calculate how much perspectives converged from their initial differences
        initial_spreads = []
        final_spreads = []

        for semantic in semantics:
            # Get initial competencies for this semantic
            initial_values = []
            final_values = []

            for perspective_obj in Perspectives:
                initial_comp = float(perspective_obj.semantic_profile.semantics[semantic])
                initial_values.append(initial_comp)

                # Get final EMA
                persp_data = df[(df['perspective_name'] == perspective_obj.name) &
                               (df['semantic_name'] == semantic)]
                if len(persp_data) > 0:
                    final_ema = persp_data['slow_ema'].iloc[-1]
                    if not pd.isna(final_ema):
                        # Combine initial competency with final adaptation
                        combined_final = initial_comp + float(final_ema)
                        final_values.append(combined_final)
                    else:
                        final_values.append(initial_comp)
                else:
                    final_values.append(initial_comp)

            initial_spread = np.std(initial_values)
            final_spread = np.std(final_values)

            initial_spreads.append(initial_spread)
            final_spreads.append(final_spread)

        avg_initial_spread = np.mean(initial_spreads)
        avg_final_spread = np.mean(final_spreads)
        convergence_factor = (avg_initial_spread - avg_final_spread) / avg_initial_spread

        print(f"Average initial spread: {avg_initial_spread:.3f}")
        print(f"Average final spread: {avg_final_spread:.3f}")
        print(f"Convergence factor: {convergence_factor:.3f} ({convergence_factor*100:.1f}%)")

        if convergence_factor > 0:
            print("→ Perspectives CONVERGED through coupled dynamics")
        else:
            print("→ Perspectives DIVERGED through competitive dynamics")

        print("="*70)

def main():
    """Generate coupled strategy visualizations"""
    viz = CoupledVisualizationAnalyzer()

    print("Generating alliance formation analysis...")
    viz.plot_alliance_formation_matrix()

    print("\nGenerating competency overlap clusters...")
    viz.plot_competency_overlap_clusters()

    print("\nGenerating halo effect analysis...")
    viz.plot_halo_effect_analysis()

    print("\nGenerating synchronization dynamics...")
    viz.plot_synchronization_dynamics()

    print("\nGenerating comprehensive coupled analysis...")
    viz.analyze_coupled_emergence_patterns()

    print("\nCoupled visualization analysis complete!")

if __name__ == "__main__":
    main()