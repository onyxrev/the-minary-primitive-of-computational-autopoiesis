# Minary: A Computational Autopoiesis Simulation

A reference implementation of the Minary computational primitive, demonstrating **autopoietic probability** through wave-like superposition consensus among multiple perspectives.

This codebase accompanies the paper:

> **"The Minary Primitive of Computational Autopoiesis"**
> Daniel Connor and Colin Defant
> [arXiv link forthcoming]

## What This Code Does

This simulation implements a discrete-time stochastic system where multiple "perspectives" evaluate "semantic dimensions" and form consensus through linear superposition. The key property: **the system learns from itself, not from external signals**. Input signals perturb the system but mathematically cancel out of the learning dynamics, creating a self-referential feedback loop.

If you're coming from a code-first angle and want to understand the theory without reading the full paper, this README maps the implementation to the core mathematical concepts.

## Conceptual Overview

### The Autopoietic Property

Traditional probabilistic systems (Bayesian networks, neural networks) are *allopoietic*: they're shaped by external ground truth. Minary is *autopoietic*: it produces its own organizational identity through internal dynamics alone.

The critical insight is **signal cancellation** (paper Equation 8): when computing learning updates, the input signal terms cancel out. What remains are only the competency-based differences between perspectives. The system learns the structure of its participants, not the content of its inputs.

### Core Components

| Paper Notation | Code Location | Description |
|----------------|---------------|-------------|
| Perspectives p₁...pₙ | `perspectives.py` | Fixed evaluators with hidden competencies |
| Semantic dimensions s₁...sₘ | `SemanticProfile.semantics` | The 19 evaluation criteria |
| Competency matrix C | `SemanticProfile.semantics` values | How good each perspective is at each dimension |
| Signal x⁽ᵗ⁾ | `Engine._signal_()` | Random input from environment (uniform [0,1]) |
| EMA memory Δ⁽ᵗ⁾ | `semantic_adjustments` | The emergent "identity" of the system |
| Consensus G⁽ᵗ⁾ | `Consensus._superposition_()` | Collective output |
| Step size α | `SLOW_EMA_ALPHA` (0.02) | Learning rate for EMA updates |

## How the Math Maps to Code

### The Response Calculation (Paper Equations 1-3)

Each perspective generates a response by comparing the input signal to its competency:

```
Raw response:      r⁽ᵗ⁾ᵢ,ⱼ = x⁽ᵗ⁾ⱼ - Cᵢ,ⱼ
Adjusted response: R⁽ᵗ⁾ᵢ,ⱼ = r⁽ᵗ⁾ᵢ,ⱼ + Δ⁽ᵗ⁻¹⁾ᵢ,ⱼ
```

In code (`consensus.py`):
```python
def _raw_response_for_semantic_(self, signal_component, competency):
    return signal_component - competency  # Equation 1

def _adjusted_response_for_semantic_(self, perspective, signal_component, semantic):
    raw_response = self._raw_response_for_semantic_(signal_component, competency)
    differential = self._differential_for_semantic_(perspective, semantic_name)  # Δ⁽ᵗ⁻¹⁾
    return raw_response + differential  # Equation 2
```

### Consensus Formation (Paper Equation 4)

Responses are combined via linear superposition (sum), not multiplication:

```
G⁽ᵗ⁾ = Σᵢ R⁽ᵗ⁾ᵢ
```

In code (`consensus.py`):
```python
def _superposition_(self, responses):
    return [sum(column) for column in zip(*responses.values())]  # Equation 4
```

The normalized consensus divides by number of perspectives: `G̅⁽ᵗ⁾ = G⁽ᵗ⁾/n`

### The Learning Signal (Paper Equation 6)

Each perspective compares its response to the collective:

```
d⁽ᵗ⁾ᵢ = G̅⁽ᵗ⁾ - R⁽ᵗ⁾ᵢ
```

In code (`consensus.py`):
```python
def _adjustment_differential_(self, i, normalized_consensus, response):
    return normalized_consensus[i] - response[i]  # Equation 6
```

**Critical property**: The paper proves (Equation 8) that `Σᵢ d⁽ᵗ⁾ᵢ = 0` exactly. Learning signals sum to zero, making this a zero-sum closed system.

### EMA Update (Paper Equation 5)

The memory matrix updates via exponential moving average:

```
Δ⁽ᵗ⁾ᵢ,ⱼ = α·d⁽ᵗ⁾ᵢ + (1-α)·Δ⁽ᵗ⁻¹⁾ᵢ,ⱼ   (for active dimensions)
```

In code (`data_types.py`):
```python
def calculate_exponential_moving_average(self, old, new, alpha):
    return (new * alpha) + (old * (1 - alpha))  # Equation 5
```

## Key Simulation Modes

### Consensus Strategy

**Superposition** (default): Responses sum linearly, preserving information through constructive/destructive interference in range [-1, 1].

**Product**: Responses multiply (after transformation to [0, 1]), collapsing information. Included for comparison with traditional probabilistic methods.

### Dimension Strategy

**Independent**: Each semantic dimension evolves separately. Simpler dynamics, dimensions are orthogonal.

**Coupled**: Perspectives average their responses across all active dimensions before contributing to consensus. This creates cross-dimensional interactions and emergent phenomena like the "halo effect" described in Section 7 of the paper.

The paper's worked examples use coupled mode with k=3 active dimensions per iteration.

## Quick Start

```bash
# Install dependencies
pip install pandas matplotlib numpy scipy seaborn

# Run basic simulation (100k iterations, superposition consensus)
python main.py

# Run the paper's worked example configuration
python main.py \
  --consensus-strategy superposition \
  --dimensions-strategy coupled \
  --dimensions-concurrency 3 \
  --iterations-count-max 10000 \
  --csv-export-enabled

# Generate visualizations
python visualize.py
python visualize_coupled.py
python visualize_consensus.py
```

## Understanding the Output

### The EMA Matrix (Δ⁽ᵗ⁾)

This is the system's emergent identity. After many iterations, it converges to a stationary distribution (Theorem 5.1 in the paper). The limiting expectation depends only on competency structure:

```
lim E[Δ⁽ᵗ⁾ᵢ,ⱼ] = (1/2 - η)(C̄ᵢ,· - C̄̄) + η(Cᵢ,ⱼ - C̄·,ⱼ)
```

Where η depends on the number of active dimensions k and total dimensions m.

### Consensus Distribution (Theorem 6.1)

The paper derives exact formulas for the mean and variance of consensus conditioned on which dimension is active. The key insight: consensus statistics depend on competency structure, not on input signal distribution.

## Perspective Archetypes

The simulation includes 5 predefined perspectives (`perspectives.py`), each with a 19-dimensional competency profile:

| Perspective | Character | Strengths |
|-------------|-----------|-----------|
| The True Artist | Creative visionary | illustration, modern art, character design |
| The Executive Director | Business leader | brand voice, visual ads, audience relevance |
| The Technician | Technical expert | 3D modeling, color grading, physics |
| The Critic | Analytical evaluator | sentiment, fashion trends, artwork similarity |
| The Fan | Audience proxy | audience relevance, sentiment, usability |

These create the competency asymmetries that drive emergent behaviors.

## Emergent Phenomena

The paper (Section 7) describes several emergent behaviors visible in simulation:

1. **Signal Cancellation**: Input signals affect responses but completely cancel from learning dynamics.

2. **Information Conservation**: Learning signals sum to zero (`Σᵢ d⁽ᵗ⁾ᵢ = 0`), unlike multiplicative systems that collapse toward zero.

3. **The Halo Effect**: In coupled mode, expertise in one dimension can propagate to influence others (Section 7.4-7.5).

4. **Promotion of the Generalist**: When dimensions are coupled, moderate competencies across all dimensions can outperform specialists (Section 7.4).

## File Structure

```
├── main.py                # CLI entry point
├── engine.py              # Simulation loop, iteration logic
├── consensus.py           # Response calculation, consensus formation, EMA updates
├── data_types.py          # Core data structures (Perspective, Iteration, etc.)
├── perspectives.py        # The 5 perspective archetypes with competency profiles
├── persistence.py         # State save/load for long runs
├── csv_export.py          # Data export for analysis
├── visualize.py           # Core analysis plots
├── visualize_consensus.py # Consensus dynamics plots
├── visualize_coupled.py   # Coupled-mode analysis
├── test_engine.py         # Engine tests
└── test_consensus.py      # Consensus calculation tests
```

## Command Line Options

```bash
python main.py [options]

--consensus-strategy superposition|product (default: superposition)
--dimensions-strategy independent|coupled (default: coupled)
--dimensions-concurrency N (default: 3)
--perspectives-count N (default: all 5)
--iterations-count-max N (default: 100,000)
--random-seed N (for reproducibility)
--persistence-enabled (save state every 10k iterations)
--csv-export-enabled (export data to export/ directory)
```

## Tests

```bash
python test_engine.py
python test_consensus.py
```

Tests verify mathematical correctness, deterministic behavior with fixed seeds, and component integration.

## Paper Reference

For the full mathematical treatment including:
- Convergence proofs using iterated random function theory (Section 5)
- Exact formulas for limiting expectations (Theorem 5.1)
- Consensus distribution analysis (Theorem 6.1)
- Detailed worked examples (Section 7)
- The argument for autopoiesis (Sections 8-9)

See the accompanying paper on arXiv.

## Citation

```bibtex
@article{connor2025minary,
  title={The Minary Primitive of Computational Autopoiesis},
  author={Connor, Daniel and Defant, Colin},
  journal={arXiv preprint},
  year={2026}
}
```

## Tools Disclosure

The core engine and consensus code were written by a human. Visualization, CSV export, and persistence code were written with AI assistance.

## License

This code is made available under the **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) License.

Copyright 2026 Autopoetic.

The Minary computational primitive is patent-pending by Autopoetic.
Autopoetic is a registered trademark of Dan Connor Consulting LLC.
