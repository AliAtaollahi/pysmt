#
# This file is part of pySMT.
#
#   Copyright 2026
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import unittest

from pysmt.environment import get_env
from pysmt.logics import QF_ALIA, QF_LIA, QF_LRA, QF_UFLIA
from pysmt.shortcuts import (
    And,
    Equals,
    Function,
    Int,
    LE,
    Not,
    Real,
    Select,
    Solver,
    Symbol,
    binary_interpolant,
    is_unsat,
    sequence_interpolant,
)
from pysmt.test import TestCase, main
from pysmt.typing import ArrayType, FunctionType, INT, REAL


HAS_SMTINTERPOL = "smtinterpol" in get_env().factory.all_solvers()


@unittest.skipUnless(HAS_SMTINTERPOL, "SMTInterpol is not available")
class TestSMTInterpol(TestCase):

    def assert_interpolant(self, a, b, interpolant, logic):
        self.assertIsNotNone(interpolant)
        self.assertTrue(
            is_unsat(
                And(a, Not(interpolant)),
                solver_name="smtinterpol",
                logic=logic,
            )
        )
        self.assertTrue(
            is_unsat(
                And(interpolant, b),
                solver_name="smtinterpol",
                logic=logic,
            )
        )

    def test_incremental_solver(self):
        x = Symbol("smtinterpol_solver_x", INT)
        zero = Equals(x, Int(0))

        with Solver(name="smtinterpol", logic=QF_LIA) as solver:
            solver.add_assertion(zero)
            self.assertTrue(solver.solve())
            self.assertEqual(solver.get_value(x), Int(0))
            solver.push()
            solver.add_assertion(Equals(x, Int(1)))
            self.assertFalse(solver.solve())
            solver.pop()
            self.assertTrue(solver.solve())

    def test_binary_and_sequence_lia(self):
        x = Symbol("smtinterpol_lia_x", INT)
        a = Equals(x, Int(0))
        middle = LE(x, Int(0))
        b = Equals(x, Int(1))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name="smtinterpol",
            logic=QF_LIA,
        )
        self.assert_interpolant(a, b, interpolant, QF_LIA)

        sequence = sequence_interpolant(
            [a, middle, b],
            solver_name="smtinterpol",
            logic=QF_LIA,
        )
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 2)
        self.assert_interpolant(a, And(middle, b), sequence[0], QF_LIA)
        self.assert_interpolant(
            And(sequence[0], middle),
            b,
            sequence[1],
            QF_LIA,
        )

    def test_binary_lra(self):
        x = Symbol("smtinterpol_lra_x", REAL)
        a = LE(x, Real(0))
        b = LE(Real(1), x)

        interpolant = binary_interpolant(
            a,
            b,
            solver_name="smtinterpol",
            logic=QF_LRA,
        )
        self.assert_interpolant(a, b, interpolant, QF_LRA)

    def test_array_interpolation(self):
        array = Symbol(
            "smtinterpol_array",
            ArrayType(INT, INT),
        )
        index = Symbol("smtinterpol_index", INT)
        selected = Select(array, index)
        a = Equals(selected, Int(0))
        b = Equals(selected, Int(1))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name="smtinterpol",
            logic=QF_ALIA,
        )
        self.assert_interpolant(a, b, interpolant, QF_ALIA)

    def test_uninterpreted_function_interpolation(self):
        x = Symbol("smtinterpol_uf_x", INT)
        function = Symbol(
            "smtinterpol_uf",
            FunctionType(INT, [INT]),
        )
        application = Function(function, [x])
        a = Equals(application, Int(0))
        b = Equals(application, Int(1))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name="smtinterpol",
            logic=QF_UFLIA,
        )
        self.assert_interpolant(a, b, interpolant, QF_UFLIA)

    def test_quoted_symbol_identity_is_preserved(self):
        state = Symbol("|smtinterpol state|", INT)
        a = Equals(state, Int(0))
        b = Equals(state, Int(1))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name="smtinterpol",
            logic=QF_LIA,
        )
        self.assert_interpolant(a, b, interpolant, QF_LIA)
        self.assertIn(state, interpolant.get_free_variables())

    def test_sat_problem_has_no_interpolant(self):
        x = Symbol("smtinterpol_sat_x", INT)
        formula = Equals(x, Int(0))
        interpolant = binary_interpolant(
            formula,
            formula,
            solver_name="smtinterpol",
            logic=QF_LIA,
        )
        self.assertIsNone(interpolant)


if __name__ == "__main__":
    main()
