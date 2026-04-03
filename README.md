# Symbolically Bounded Latent Recursion

## Overview

This repository fuses a **Tiny Recursive Model (TRM)** with a deterministic **Z3 SMT solver** at test-time to mathematically prevent representation collapse and hallucination in deep latent reasoning. The TRM is a lightweight, weight-shared transformer that solves constraint-satisfaction problems (Sudoku, Mazes) through iterative latent recursion. By intercepting the model's predictions with a formal Z3 solver during inference, we enforce hard symbolic constraints that the neural network alone cannot guarantee.

## Project Structure

```
├── data/                 # Dataset generation modules
│   ├── __init__.py       # Dataset factory (get_dataset)
│   ├── sudoku.py         # Sudoku puzzle generation & dataset
│   └── maze.py           # Maze generation (randomised DFS) & dataset
├── viz/                  # Visualization utilities
│   ├── __init__.py       # Public API re-exports
│   └── visualize.py      # Console + matplotlib visualizations
├── symbolic/             # Neuro-symbolic integration (Z3)
│   ├── __init__.py
│   └── z3_sudoku.py      # Z3 test-time intercept (TODO)
├── model.py              # TRM architecture (TaskHeads + recursive core)
├── train.py              # Training loop with EMA & deep supervision
├── evaluate.py           # Task-specific evaluation metrics
├── device.py             # Device utilities (MPS / Apple Silicon)
├── check_gpu.py          # Quick GPU availability check
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/usafhulk/Symbolically-Bounded-Latent-Recursion-.git
cd Symbolically-Bounded-Latent-Recursion-

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Check GPU availability
python check_gpu.py

# Train on Sudoku
python train.py --task sudoku

# Train on Maze
python train.py --task maze --grid_size 11
```