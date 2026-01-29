#!/usr/bin/env python3
"""
Polarity-Aware SAT Solver

A SAT solver that uses polarity information to guide decision making.
The polarity of a literal is the difference between its positive and negative occurrences
in the formula. This information can be used to make better branching decisions.
"""

from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict


class CNFFormula:
    """Represents a CNF (Conjunctive Normal Form) formula."""
    
    def __init__(self, clauses: List[List[int]]):
        """
        Initialize CNF formula.
        
        Args:
            clauses: List of clauses, where each clause is a list of literals.
                    A literal is a non-zero integer (positive or negative).
                    Variable n is represented by n, and its negation by -n.
        """
        self.clauses = [clause[:] for clause in clauses]  # Deep copy
        self.original_clauses = [clause[:] for clause in clauses]
        self.num_variables = self._compute_num_variables()
    
    def _compute_num_variables(self) -> int:
        """Compute the number of variables in the formula."""
        variables = set()
        for clause in self.clauses:
            for lit in clause:
                variables.add(abs(lit))
        return len(variables)
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """Check if the formula is satisfied by the given assignment."""
        for clause in self.original_clauses:
            clause_satisfied = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                return False
        return True


class PolarityAwareSATSolver:
    """
    A DPLL-based SAT solver with polarity-aware branching heuristic.
    
    The polarity heuristic prefers to assign variables based on their
    predominant polarity in the formula.
    """
    
    def __init__(self, formula: CNFFormula):
        """Initialize the solver with a CNF formula."""
        self.formula = formula
        self.assignment: Dict[int, bool] = {}
        self.polarity_scores: Dict[int, int] = {}
        self._compute_polarity_scores()
    
    def _compute_polarity_scores(self) -> None:
        """
        Compute polarity scores for all variables.
        
        Polarity score = (positive occurrences) - (negative occurrences)
        Positive score means variable appears more often positive,
        negative score means it appears more often negative.
        """
        self.polarity_scores = defaultdict(int)
        
        for clause in self.formula.original_clauses:
            for lit in clause:
                var = abs(lit)
                if lit > 0:
                    self.polarity_scores[var] += 1
                else:
                    self.polarity_scores[var] -= 1
    
    def solve(self) -> Optional[Dict[int, bool]]:
        """
        Solve the SAT problem.
        
        Returns:
            A satisfying assignment if SAT, None if UNSAT.
        """
        result = self._dpll(self.formula.clauses[:], {})
        
        if result is not None:
            # Verify the solution
            if not self.formula.is_satisfied(result):
                raise RuntimeError("Invalid solution found!")
        
        return result
    
    def _dpll(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        """
        DPLL algorithm with polarity-aware branching.
        
        Args:
            clauses: Current set of clauses
            assignment: Current partial assignment
        
        Returns:
            A satisfying assignment if SAT, None if UNSAT.
        """
        # Check for empty clause (UNSAT)
        if any(len(clause) == 0 for clause in clauses):
            return None
        
        # Check if all clauses satisfied (SAT)
        if len(clauses) == 0:
            return assignment.copy()
        
        # Unit propagation
        clauses, assignment = self._unit_propagate(clauses, assignment)
        
        if clauses is None:
            return None
        
        if len(clauses) == 0:
            return assignment.copy()
        
        # Pure literal elimination
        clauses, assignment = self._pure_literal_eliminate(clauses, assignment)
        
        if len(clauses) == 0:
            return assignment.copy()
        
        # Choose variable using polarity-aware heuristic
        var = self._choose_variable_polarity_aware(clauses, assignment)
        
        # Try polarity-preferred value first
        preferred_value = self._get_preferred_polarity(var)
        
        # Try preferred polarity first
        new_assignment = assignment.copy()
        new_assignment[var] = preferred_value
        new_clauses = self._simplify(clauses, var, preferred_value)
        
        result = self._dpll(new_clauses, new_assignment)
        if result is not None:
            return result
        
        # Try opposite polarity
        new_assignment = assignment.copy()
        new_assignment[var] = not preferred_value
        new_clauses = self._simplify(clauses, var, not preferred_value)
        
        return self._dpll(new_clauses, new_assignment)
    
    def _unit_propagate(self, clauses: List[List[int]], 
                        assignment: Dict[int, bool]) -> Tuple[Optional[List[List[int]]], Dict[int, bool]]:
        """
        Perform unit propagation.
        
        Returns:
            Tuple of (simplified clauses, updated assignment) or (None, assignment) if conflict.
        """
        assignment = assignment.copy()
        clauses = [clause[:] for clause in clauses]
        
        changed = True
        while changed:
            changed = False
            unit_clauses = [clause for clause in clauses if len(clause) == 1]
            
            for unit_clause in unit_clauses:
                lit = unit_clause[0]
                var = abs(lit)
                value = lit > 0
                
                if var in assignment:
                    if assignment[var] != value:
                        return None, assignment
                    continue
                
                assignment[var] = value
                clauses = self._simplify(clauses, var, value)
                
                if any(len(clause) == 0 for clause in clauses):
                    return None, assignment
                
                changed = True
                break
        
        return clauses, assignment
    
    def _pure_literal_eliminate(self, clauses: List[List[int]], 
                                assignment: Dict[int, bool]) -> Tuple[List[List[int]], Dict[int, bool]]:
        """
        Eliminate pure literals (variables that appear with only one polarity).
        
        Returns:
            Tuple of (simplified clauses, updated assignment).
        """
        assignment = assignment.copy()
        clauses = [clause[:] for clause in clauses]
        
        # Find all literals in remaining clauses
        positive_vars = set()
        negative_vars = set()
        
        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    positive_vars.add(lit)
                else:
                    negative_vars.add(-lit)
        
        # Find pure literals
        pure_positive = positive_vars - negative_vars
        pure_negative = negative_vars - positive_vars
        
        # Assign pure literals
        for var in pure_positive:
            if var not in assignment:
                assignment[var] = True
                clauses = self._simplify(clauses, var, True)
        
        for var in pure_negative:
            if var not in assignment:
                assignment[var] = False
                clauses = self._simplify(clauses, var, False)
        
        return clauses, assignment
    
    def _simplify(self, clauses: List[List[int]], var: int, value: bool) -> List[List[int]]:
        """
        Simplify clauses given a variable assignment.
        
        Args:
            clauses: List of clauses to simplify
            var: Variable to assign
            value: Value to assign to variable (True/False)
        
        Returns:
            Simplified list of clauses.
        """
        new_clauses = []
        lit = var if value else -var
        
        for clause in clauses:
            if lit in clause:
                # Clause is satisfied, remove it
                continue
            
            # Remove negation of lit from clause
            new_clause = [l for l in clause if l != -lit]
            new_clauses.append(new_clause)
        
        return new_clauses
    
    def _choose_variable_polarity_aware(self, clauses: List[List[int]], 
                                        assignment: Dict[int, bool]) -> int:
        """
        Choose next variable to branch on using polarity-aware heuristic.
        
        Uses a combination of:
        1. Most frequent variable in remaining clauses
        2. Polarity information
        
        Returns:
            Variable to branch on.
        """
        var_freq = defaultdict(int)
        
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    var_freq[var] += 1
        
        if not var_freq:
            # All variables assigned
            return 1
        
        # Choose most frequent variable
        return max(var_freq.keys(), key=lambda v: var_freq[v])
    
    def _get_preferred_polarity(self, var: int) -> bool:
        """
        Get preferred polarity for a variable based on its polarity score.
        
        Args:
            var: Variable to get polarity for
        
        Returns:
            True if positive polarity preferred, False if negative polarity preferred.
        """
        score = self.polarity_scores.get(var, 0)
        return score >= 0  # Prefer positive if score >= 0


def parse_dimacs(text: str) -> CNFFormula:
    """
    Parse a CNF formula in DIMACS format.
    
    Args:
        text: DIMACS format text
    
    Returns:
        CNFFormula object
    """
    clauses = []
    
    for line in text.strip().split('\n'):
        line = line.strip()
        
        # Skip comments and problem line
        if line.startswith('c') or line.startswith('p'):
            continue
        
        if not line:
            continue
        
        # Parse clause
        literals = [int(x) for x in line.split()]
        
        # Remove trailing 0
        if literals and literals[-1] == 0:
            literals = literals[:-1]
        
        if literals:
            clauses.append(literals)
    
    return CNFFormula(clauses)


def solve_sat(clauses: List[List[int]]) -> Optional[Dict[int, bool]]:
    """
    Convenience function to solve a SAT problem.
    
    Args:
        clauses: List of clauses in CNF format
    
    Returns:
        Satisfying assignment if SAT, None if UNSAT
    """
    formula = CNFFormula(clauses)
    solver = PolarityAwareSATSolver(formula)
    return solver.solve()


if __name__ == "__main__":
    # Example usage
    print("Polarity-Aware SAT Solver")
    print("=" * 50)
    
    # Example 1: Simple satisfiable formula
    # (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
    clauses1 = [
        [1, 2],      # x1 ∨ x2
        [-1, 3],     # ¬x1 ∨ x3
        [-2, -3]     # ¬x2 ∨ ¬x3
    ]
    
    print("\nExample 1: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)")
    result1 = solve_sat(clauses1)
    if result1:
        print(f"SAT - Solution: {result1}")
    else:
        print("UNSAT")
    
    # Example 2: Unsatisfiable formula
    # (x1) ∧ (¬x1)
    clauses2 = [
        [1],         # x1
        [-1]         # ¬x1
    ]
    
    print("\nExample 2: (x1) ∧ (¬x1)")
    result2 = solve_sat(clauses2)
    if result2:
        print(f"SAT - Solution: {result2}")
    else:
        print("UNSAT")
    
    # Example 3: Larger satisfiable formula
    # (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2) ∧ (¬x1 ∨ ¬x3) ∧ (¬x2 ∨ ¬x3)
    clauses3 = [
        [1, 2, 3],   # x1 ∨ x2 ∨ x3
        [-1, -2],    # ¬x1 ∨ ¬x2
        [-1, -3],    # ¬x1 ∨ ¬x3
        [-2, -3]     # ¬x2 ∨ ¬x3
    ]
    
    print("\nExample 3: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2) ∧ (¬x1 ∨ ¬x3) ∧ (¬x2 ∨ ¬x3)")
    result3 = solve_sat(clauses3)
    if result3:
        print(f"SAT - Solution: {result3}")
    else:
        print("UNSAT")
