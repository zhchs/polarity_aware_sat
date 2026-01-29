# Polarity-Aware SAT Solver

A Boolean Satisfiability (SAT) solver that uses polarity information to guide decision making. The solver implements the DPLL algorithm enhanced with a polarity-aware branching heuristic.

## Overview

This SAT solver uses the polarity of literals (the difference between positive and negative occurrences) to make smarter branching decisions. When a variable appears more often with positive polarity, the solver prefers to try assigning it `True` first, and vice versa for negative polarity.

## Features

- **DPLL Algorithm**: Classic backtracking search with unit propagation and pure literal elimination
- **Polarity-Aware Heuristic**: Branching decisions guided by literal polarity in the formula
- **CNF Format Support**: Handles formulas in Conjunctive Normal Form
- **DIMACS Parser**: Can parse standard DIMACS format input
- **Comprehensive Testing**: Full test suite with various SAT/UNSAT problems

## Installation

No external dependencies required! The solver uses only Python standard library.

```bash
# Clone the repository
git clone https://github.com/zhchs/polarity_aware_sat.git
cd polarity_aware_sat

# No installation needed - just run!
python polarity_aware_sat.py
```

## Usage

### Basic Usage

```python
from polarity_aware_sat import solve_sat, CNFFormula, PolarityAwareSATSolver

# Define a CNF formula as a list of clauses
# Each clause is a list of literals (non-zero integers)
# Positive integer = variable, Negative integer = negation
clauses = [
    [1, 2],      # x1 ∨ x2
    [-1, 3],     # ¬x1 ∨ x3
    [-2, -3]     # ¬x2 ∨ ¬x3
]

# Solve the formula
result = solve_sat(clauses)

if result:
    print(f"SAT - Solution: {result}")
    # Output: SAT - Solution: {1: True, 2: False, 3: True}
else:
    print("UNSAT")
```

### Using the Solver Class Directly

```python
from polarity_aware_sat import CNFFormula, PolarityAwareSATSolver

# Create a CNF formula
formula = CNFFormula([[1, 2], [-1, 3], [-2, -3]])

# Initialize solver
solver = PolarityAwareSATSolver(formula)

# Check polarity scores
print(solver.polarity_scores)  # {1: 0, 2: 0, 3: 0}

# Solve
solution = solver.solve()
print(solution)
```

### Parsing DIMACS Format

```python
from polarity_aware_sat import parse_dimacs, PolarityAwareSATSolver

dimacs_text = """
c This is a comment
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
"""

formula = parse_dimacs(dimacs_text)
solver = PolarityAwareSATSolver(formula)
result = solver.solve()
```

## Running Examples

The main script includes several examples:

```bash
python polarity_aware_sat.py
```

Output:
```
Polarity-Aware SAT Solver
==================================================

Example 1: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
SAT - Solution: {1: True, 2: False, 3: True}

Example 2: (x1) ∧ (¬x1)
UNSAT

Example 3: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2) ∧ (¬x1 ∨ ¬x3) ∧ (¬x2 ∨ ¬x3)
SAT - Solution: {1: True, 2: False, 3: False}
```

## Running Tests

Run the comprehensive test suite:

```bash
python -m pytest test_sat_solver.py -v
```

Or using unittest:

```bash
python test_sat_solver.py
```

## How It Works

### Polarity Scoring

The solver computes a polarity score for each variable:
- **Polarity Score** = (positive occurrences) - (negative occurrences)
- Positive score → variable appears more often positive → prefer `True`
- Negative score → variable appears more often negative → prefer `False`

### DPLL Algorithm

1. **Unit Propagation**: Assign values to unit clauses (clauses with single literal)
2. **Pure Literal Elimination**: Assign values to variables appearing with only one polarity
3. **Branching**: Choose unassigned variable and try both values
4. **Backtracking**: If conflict found, try alternative branch

### Polarity-Aware Branching

When branching on a variable, the solver:
1. Selects the most frequent variable in remaining clauses
2. Tries the polarity-preferred value first (based on polarity score)
3. Backtracks and tries opposite value if needed

## Algorithm Complexity

- **Best Case**: O(n) with unit propagation and pure literal elimination
- **Worst Case**: O(2^n) for hard instances (exponential backtracking)
- **Space**: O(n × m) where n = variables, m = clauses

## Examples

### Satisfiable Formula
```python
# (x1 ∨ x2) ∧ (¬x1 ∨ x3)
clauses = [[1, 2], [-1, 3]]
result = solve_sat(clauses)
# Result: {1: True, 2: True, 3: True} (one possible solution)
```

### Unsatisfiable Formula
```python
# (x1) ∧ (¬x1)
clauses = [[1], [-1]]
result = solve_sat(clauses)
# Result: None (UNSAT)
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests

## License

This project is open source and available under the MIT License.

## References

- [DPLL Algorithm](https://en.wikipedia.org/wiki/DPLL_algorithm)
- [Boolean Satisfiability Problem](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)
- [DIMACS CNF Format](https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps)