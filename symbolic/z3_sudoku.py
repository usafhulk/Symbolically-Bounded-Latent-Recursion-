from z3 import *
import torch


def solve_4x4_sudoku(givens):
    """
    Solves a 4x4 mini-Sudoku using the Z3 SMT solver.
    'givens' is a 4x4 list of lists, where 0 represents an empty cell.
    """
    # 1. THE VARIABLES (The Latent Space)
    # Create a 4x4 matrix of Z3 Integer variables
    # X[i][j] represents the value in row i, column j
    X = [[Int(f'x_{i}_{j}') for j in range(4)] for i in range(4)]

    # Initialize the solver
    s = Solver()

    # 2. THE CONSTRAINTS (The Logical Guardrails)

    # A. Cell value constraints (Must be 1, 2, 3, or 4)
    for i in range(4):
        for j in range(4):
            s.add(And(X[i][j] >= 1, X[i][j] <= 4))

    # B. Row constraints (Numbers in each row must be distinct)
    for i in range(4):
        s.add(Distinct(X[i]))

    # C. Column constraints (Numbers in each column must be distinct)
    for j in range(4):
        col = [X[i][j] for i in range(4)]
        s.add(Distinct(col))

    # D. Subgrid constraints (Numbers in each 2x2 box must be distinct)
    for i0 in range(2):
        for j0 in range(2):
            subgrid = [X[2*i0 + i][2*j0 + j]
                       for i in range(2) for j in range(2)]
            s.add(Distinct(subgrid))

    # 3. APPLYING THE GIVENS (Or your Neural Network's guess)
    for i in range(4):
        for j in range(4):
            if givens[i][j] != 0:
                s.add(X[i][j] == givens[i][j])

    # 4. EVALUATION
    if s.check() == sat:
        print("SATISFIABLE: Valid state found!")
        m = s.model()
        # Extract the solution back into a standard Python list
        solution = [[m.evaluate(X[i][j]).as_long()
                     for j in range(4)] for i in range(4)]

        # Print the grid beautifully
        print("-" * 13)
        for i in range(4):
            row_str = "| " + " ".join(str(solution[i][j]) for j in range(2)) + " | " + \
                      " ".join(str(solution[i][j]) for j in range(2, 4)) + " |"
            print(row_str)
            if i % 2 == 1:
                print("-" * 13)
        return solution
    else:
        print("UNSATISFIABLE: This grid state violates the rules.")
        return None


# --- Test the Solver ---
if __name__ == "__main__":
    # A valid 4x4 Sudoku with some empty cells (0)
    test_grid = [
        [0, 3, 4, 0],
        [4, 0, 0, 2],
        [1, 0, 0, 3],
        [0, 2, 1, 0]
    ]

    print("Testing valid grid...")
    solve_4x4_sudoku(test_grid)

    # An invalid grid (hallucinating two 4s in the top row)
    invalid_grid = [
        [4, 3, 4, 0],
        [4, 0, 0, 2],
        [1, 0, 0, 3],
        [0, 2, 1, 0]
    ]

    print("\nTesting invalid grid (Neuro-Symbolic Intercept Simulation)...")
    solve_4x4_sudoku(invalid_grid)


# ---------------------------------------------------------------------------
# Neuro-Symbolic Logit Pruning for 4x4 Sudoku
# ---------------------------------------------------------------------------

def _build_base_constraints_4x4():
    """Build the Z3 variables and static 4x4 Sudoku constraints once.

    Returns (X, base_constraints) where:
        X: 4x4 list of Z3 Int variables
        base_constraints: list of Z3 constraints (range, row, col, box)
    """
    X = [[Int(f'x_{i}_{j}') for j in range(4)] for i in range(4)]
    constraints = []

    # Cell values in [1, 4]
    for i in range(4):
        for j in range(4):
            constraints.append(And(X[i][j] >= 1, X[i][j] <= 4))

    # Row uniqueness
    for i in range(4):
        constraints.append(Distinct(X[i]))

    # Column uniqueness
    for j in range(4):
        constraints.append(Distinct([X[i][j] for i in range(4)]))

    # 2x2 subgrid uniqueness
    for bi in range(2):
        for bj in range(2):
            constraints.append(Distinct([
                X[2 * bi + di][2 * bj + dj]
                for di in range(2) for dj in range(2)
            ]))

    return X, constraints


def _get_illegal_digits_4x4(board_flat, cell_idx, X, base_constraints):
    """For a single empty cell, return the set of digits that Z3 proves illegal.

    Args:
        board_flat: length-16 list of ints (0 = empty, 1-4 = filled)
        cell_idx:   flat index 0..15 of the cell to check
        X:          Z3 variable grid from _build_base_constraints_4x4
        base_constraints: static constraints list

    Returns:
        set of ints in {1,2,3,4} that are UNSAT for this cell
    """
    r, c = divmod(cell_idx, 4)
    illegal = set()

    # Build "given" constraints from currently filled cells
    given_constraints = []
    for idx in range(16):
        if board_flat[idx] != 0:
            gi, gj = divmod(idx, 4)
            given_constraints.append(X[gi][gj] == int(board_flat[idx]))

    for digit in range(1, 5):
        s = Solver()
        s.add(base_constraints)
        s.add(given_constraints)
        s.add(X[r][c] == digit)
        if s.check() == unsat:
            illegal.add(digit)

    return illegal


def prune_illegal_logits_z3(y_hat, current_hard_predictions):
    """Prune logits for a 4x4 Sudoku using Z3 constraint checking.

    For every empty cell, Z3 checks which digits are mathematically impossible
    given the current board state. Illegal digit logits are set to -1e9.

    This function operates on CPU tensors. If inputs are on GPU/MPS, they are
    moved to CPU for Z3 processing and the result is moved back.

    Args:
        y_hat: [B, 16, 4] raw logits (digits 1-4 mapped to indices 0-3)
        current_hard_predictions: [B, 16] discrete board (0=empty, 1-4=filled)

    Returns:
        y_hat: [B, 16, 4] with illegal logits masked to -1e9 (same device as input)
    """
    src_device = y_hat.device
    y_hat = y_hat.detach().clone().to('cpu')
    preds = current_hard_predictions.detach().to('cpu')

    B = y_hat.shape[0]
    X, base_constraints = _build_base_constraints_4x4()

    for b in range(B):
        board = preds[b].tolist()

        for cell_idx in range(16):
            if board[cell_idx] != 0:
                continue  # skip filled cells

            illegal = _get_illegal_digits_4x4(board, cell_idx, X, base_constraints)

            for digit in illegal:
                # digit 1-4 -> logit index 0-3
                y_hat[b, cell_idx, digit - 1] = -1e9

    return y_hat.to(src_device)
