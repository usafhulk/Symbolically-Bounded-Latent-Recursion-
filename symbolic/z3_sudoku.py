from z3 import *


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
