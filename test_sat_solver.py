#!/usr/bin/env python3
"""
Test suite for Polarity-Aware SAT Solver
"""

import unittest
from polarity_aware_sat import CNFFormula, PolarityAwareSATSolver, solve_sat, parse_dimacs


class TestCNFFormula(unittest.TestCase):
    """Tests for CNFFormula class."""
    
    def test_empty_formula(self):
        """Test empty formula (trivially satisfiable)."""
        formula = CNFFormula([])
        self.assertEqual(formula.num_variables, 0)
        self.assertTrue(formula.is_satisfied({}))
    
    def test_single_clause(self):
        """Test formula with single clause."""
        formula = CNFFormula([[1, 2]])
        self.assertEqual(formula.num_variables, 2)
        self.assertTrue(formula.is_satisfied({1: True}))
        self.assertTrue(formula.is_satisfied({2: True}))
        self.assertFalse(formula.is_satisfied({1: False, 2: False}))
    
    def test_multiple_clauses(self):
        """Test formula with multiple clauses."""
        formula = CNFFormula([[1, 2], [-1, 3], [-2, -3]])
        self.assertEqual(formula.num_variables, 3)
        self.assertTrue(formula.is_satisfied({1: True, 2: False, 3: True}))
        self.assertFalse(formula.is_satisfied({1: False, 2: False}))


class TestPolarityScores(unittest.TestCase):
    """Tests for polarity score computation."""
    
    def test_positive_polarity(self):
        """Test variable with positive polarity."""
        formula = CNFFormula([[1, 2], [1, 3], [1, 4]])
        solver = PolarityAwareSATSolver(formula)
        # Variable 1 appears 3 times positive, 0 times negative
        self.assertEqual(solver.polarity_scores[1], 3)
    
    def test_negative_polarity(self):
        """Test variable with negative polarity."""
        formula = CNFFormula([[-1, 2], [-1, 3], [-1, 4]])
        solver = PolarityAwareSATSolver(formula)
        # Variable 1 appears 0 times positive, 3 times negative
        self.assertEqual(solver.polarity_scores[1], -3)
    
    def test_mixed_polarity(self):
        """Test variable with mixed polarity."""
        formula = CNFFormula([[1, 2], [-1, 3], [1, 4]])
        solver = PolarityAwareSATSolver(formula)
        # Variable 1 appears 2 times positive, 1 time negative
        self.assertEqual(solver.polarity_scores[1], 1)


class TestSATSolver(unittest.TestCase):
    """Tests for SAT solver functionality."""
    
    def test_trivial_sat(self):
        """Test trivially satisfiable formula."""
        result = solve_sat([[1]])
        self.assertIsNotNone(result)
        self.assertTrue(result[1])
    
    def test_trivial_unsat(self):
        """Test trivially unsatisfiable formula."""
        result = solve_sat([[1], [-1]])
        self.assertIsNone(result)
    
    def test_simple_sat(self):
        """Test simple satisfiable formula."""
        # (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
        clauses = [[1, 2], [-1, 3], [-2, -3]]
        result = solve_sat(clauses)
        self.assertIsNotNone(result)
        
        # Verify solution
        formula = CNFFormula(clauses)
        self.assertTrue(formula.is_satisfied(result))
    
    def test_simple_unsat(self):
        """Test simple unsatisfiable formula."""
        # (x1 ∨ x2) ∧ (¬x1) ∧ (¬x2)
        clauses = [[1, 2], [-1], [-2]]
        result = solve_sat(clauses)
        self.assertIsNone(result)
    
    def test_unit_propagation(self):
        """Test unit propagation."""
        # (x1) ∧ (¬x1 ∨ x2) ∧ (¬x2 ∨ x3)
        clauses = [[1], [-1, 2], [-2, 3]]
        result = solve_sat(clauses)
        self.assertIsNotNone(result)
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertTrue(result[3])
    
    def test_pure_literal(self):
        """Test pure literal elimination."""
        # (x1 ∨ x2) ∧ (x1 ∨ x3) - x1 is pure positive
        clauses = [[1, 2], [1, 3]]
        result = solve_sat(clauses)
        self.assertIsNotNone(result)
        self.assertTrue(result[1])  # x1 should be True
    
    def test_larger_sat(self):
        """Test larger satisfiable formula."""
        # (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2) ∧ (¬x1 ∨ ¬x3) ∧ (¬x2 ∨ ¬x3)
        clauses = [[1, 2, 3], [-1, -2], [-1, -3], [-2, -3]]
        result = solve_sat(clauses)
        self.assertIsNotNone(result)
        
        # Verify solution
        formula = CNFFormula(clauses)
        self.assertTrue(formula.is_satisfied(result))
    
    def test_pigeonhole_3_2(self):
        """Test 3 pigeons in 2 holes (unsatisfiable)."""
        # Pigeon i in hole j: variable (i-1)*2 + j
        # Pigeon 1: vars 1, 2
        # Pigeon 2: vars 3, 4
        # Pigeon 3: vars 5, 6
        
        clauses = []
        
        # Each pigeon in at least one hole
        clauses.append([1, 2])   # Pigeon 1
        clauses.append([3, 4])   # Pigeon 2
        clauses.append([5, 6])   # Pigeon 3
        
        # At most one pigeon per hole
        # Hole 1: vars 1, 3, 5
        clauses.append([-1, -3])
        clauses.append([-1, -5])
        clauses.append([-3, -5])
        
        # Hole 2: vars 2, 4, 6
        clauses.append([-2, -4])
        clauses.append([-2, -6])
        clauses.append([-4, -6])
        
        result = solve_sat(clauses)
        self.assertIsNone(result)


class TestDIMACSParser(unittest.TestCase):
    """Tests for DIMACS format parser."""
    
    def test_parse_simple(self):
        """Test parsing simple DIMACS format."""
        dimacs = """
        c This is a comment
        p cnf 3 2
        1 2 0
        -1 3 0
        """
        formula = parse_dimacs(dimacs)
        self.assertEqual(len(formula.clauses), 2)
        self.assertEqual(formula.clauses[0], [1, 2])
        self.assertEqual(formula.clauses[1], [-1, 3])
    
    def test_parse_with_comments(self):
        """Test parsing DIMACS with multiple comments."""
        dimacs = """
        c Comment 1
        c Comment 2
        p cnf 2 2
        1 2 0
        c Another comment
        -1 -2 0
        """
        formula = parse_dimacs(dimacs)
        self.assertEqual(len(formula.clauses), 2)


class TestPolarityAwareHeuristic(unittest.TestCase):
    """Tests for polarity-aware heuristic."""
    
    def test_prefers_positive_polarity(self):
        """Test that solver prefers positive polarity when appropriate."""
        # Create formula where x1 appears more often positive
        clauses = [[1, 2], [1, 3], [1, 4], [-1, 5]]
        solver = PolarityAwareSATSolver(CNFFormula(clauses))
        
        # Variable 1 has positive polarity score (3 - 1 = 2)
        self.assertGreater(solver.polarity_scores[1], 0)
        self.assertTrue(solver._get_preferred_polarity(1))
    
    def test_prefers_negative_polarity(self):
        """Test that solver prefers negative polarity when appropriate."""
        # Create formula where x1 appears more often negative
        clauses = [[-1, 2], [-1, 3], [-1, 4], [1, 5]]
        solver = PolarityAwareSATSolver(CNFFormula(clauses))
        
        # Variable 1 has negative polarity score (1 - 3 = -2)
        self.assertLess(solver.polarity_scores[1], 0)
        self.assertFalse(solver._get_preferred_polarity(1))


if __name__ == '__main__':
    unittest.main()
