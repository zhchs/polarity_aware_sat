#!/usr/bin/env python3
"""
Advanced examples for Polarity-Aware SAT Solver
"""

from polarity_aware_sat import solve_sat, CNFFormula, PolarityAwareSATSolver, parse_dimacs


def example_3_coloring():
    """
    Graph 3-coloring problem.
    
    Given a graph, can we color each vertex with one of 3 colors
    such that no two adjacent vertices have the same color?
    
    Graph: Triangle (3 vertices, all connected)
    This is satisfiable.
    """
    print("\n" + "="*60)
    print("Example: Graph 3-Coloring (Triangle)")
    print("="*60)
    
    # Variables: vertex_i_color_j means vertex i has color j
    # Vertex 1: vars 1, 2, 3 (colors 1, 2, 3)
    # Vertex 2: vars 4, 5, 6
    # Vertex 3: vars 7, 8, 9
    
    clauses = []
    
    # Each vertex must have at least one color
    clauses.append([1, 2, 3])    # Vertex 1
    clauses.append([4, 5, 6])    # Vertex 2
    clauses.append([7, 8, 9])    # Vertex 3
    
    # Each vertex has at most one color (no two colors at once)
    # Vertex 1
    clauses.append([-1, -2])
    clauses.append([-1, -3])
    clauses.append([-2, -3])
    
    # Vertex 2
    clauses.append([-4, -5])
    clauses.append([-4, -6])
    clauses.append([-5, -6])
    
    # Vertex 3
    clauses.append([-7, -8])
    clauses.append([-7, -9])
    clauses.append([-8, -9])
    
    # Adjacent vertices have different colors
    # Edge (1, 2)
    clauses.append([-1, -4])  # If v1=color1, then v2≠color1
    clauses.append([-2, -5])  # If v1=color2, then v2≠color2
    clauses.append([-3, -6])  # If v1=color3, then v2≠color3
    
    # Edge (1, 3)
    clauses.append([-1, -7])
    clauses.append([-2, -8])
    clauses.append([-3, -9])
    
    # Edge (2, 3)
    clauses.append([-4, -7])
    clauses.append([-5, -8])
    clauses.append([-6, -9])
    
    result = solve_sat(clauses)
    
    if result:
        print("SAT - 3-coloring exists!")
        print("\nColoring:")
        for vertex in range(1, 4):
            for color in range(1, 4):
                var = (vertex - 1) * 3 + color
                if result.get(var, False):
                    print(f"  Vertex {vertex}: Color {color}")
    else:
        print("UNSAT - No 3-coloring exists")


def example_sudoku_cell():
    """
    Mini Sudoku: Single cell must have value 1-4, but not conflicting.
    Demonstrates encoding a constraint satisfaction problem.
    """
    print("\n" + "="*60)
    print("Example: Mini Sudoku Cell (1-4)")
    print("="*60)
    
    # Variables 1-4: cell has value 1, 2, 3, or 4
    clauses = []
    
    # Cell must have at least one value
    clauses.append([1, 2, 3, 4])
    
    # Cell has at most one value
    clauses.append([-1, -2])
    clauses.append([-1, -3])
    clauses.append([-1, -4])
    clauses.append([-2, -3])
    clauses.append([-2, -4])
    clauses.append([-3, -4])
    
    # Additional constraint: value must be 2 or 3 (from other cells)
    clauses.append([2, 3])
    
    result = solve_sat(clauses)
    
    if result:
        print("SAT - Valid assignment exists!")
        for val in range(1, 5):
            if result.get(val, False):
                print(f"  Cell value: {val}")
    else:
        print("UNSAT")


def example_with_polarity_analysis():
    """
    Demonstrate polarity-aware heuristic in action.
    """
    print("\n" + "="*60)
    print("Example: Polarity Analysis")
    print("="*60)
    
    # Create a formula where x1 appears mostly positive
    clauses = [
        [1, 2],
        [1, 3],
        [1, 4],
        [-1, 5],
        [-2, -3, -4]
    ]
    
    formula = CNFFormula(clauses)
    solver = PolarityAwareSATSolver(formula)
    
    print("\nPolarity scores:")
    for var in sorted(solver.polarity_scores.keys()):
        score = solver.polarity_scores[var]
        polarity = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
        print(f"  Variable {var}: score={score:+2d} ({polarity})")
    
    result = solver.solve()
    
    if result:
        print(f"\nSAT - Solution: {result}")
    else:
        print("\nUNSAT")


def example_dimacs_format():
    """
    Example using DIMACS format.
    """
    print("\n" + "="*60)
    print("Example: DIMACS Format")
    print("="*60)
    
    dimacs = """
    c Example CNF formula in DIMACS format
    c (x1 ∨ ¬x2) ∧ (x2 ∨ x3) ∧ (¬x1 ∨ ¬x3)
    p cnf 3 3
    1 -2 0
    2 3 0
    -1 -3 0
    """
    
    print("\nDIMACS input:")
    print(dimacs)
    
    formula = parse_dimacs(dimacs)
    solver = PolarityAwareSATSolver(formula)
    result = solver.solve()
    
    if result:
        print(f"SAT - Solution: {result}")
    else:
        print("UNSAT")


def example_pigeonhole():
    """
    Pigeonhole principle: n+1 pigeons in n holes.
    This is a classic UNSAT problem.
    """
    print("\n" + "="*60)
    print("Example: Pigeonhole Principle (4 pigeons, 3 holes)")
    print("="*60)
    
    n_pigeons = 4
    n_holes = 3
    
    clauses = []
    
    # Each pigeon must be in at least one hole
    for pigeon in range(n_pigeons):
        clause = []
        for hole in range(n_holes):
            var = pigeon * n_holes + hole + 1
            clause.append(var)
        clauses.append(clause)
    
    # At most one pigeon per hole
    for hole in range(n_holes):
        for p1 in range(n_pigeons):
            for p2 in range(p1 + 1, n_pigeons):
                var1 = p1 * n_holes + hole + 1
                var2 = p2 * n_holes + hole + 1
                clauses.append([-var1, -var2])
    
    print(f"\n{len(clauses)} clauses generated")
    
    result = solve_sat(clauses)
    
    if result:
        print("SAT - Assignment found (unexpected!)")
    else:
        print("UNSAT - Cannot fit 4 pigeons in 3 holes (as expected)")


if __name__ == "__main__":
    print("\nPolarity-Aware SAT Solver - Advanced Examples")
    print("="*60)
    
    example_with_polarity_analysis()
    example_3_coloring()
    example_sudoku_cell()
    example_dimacs_format()
    example_pigeonhole()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
