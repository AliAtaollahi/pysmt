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

import gc
import unittest
import weakref

from pysmt.environment import get_env
from pysmt.logics import QF_ABV, QF_BV, QF_LIA
from pysmt.solvers._smtlib_bridge import term_to_pysmt, terms_to_pysmt
from pysmt.solvers._array_rewriter import (
    ArrayEqualityExpander,
    expand_array_equalities,
)
from pysmt.shortcuts import (
    And,
    Array,
    BV,
    Equals,
    GE,
    Int,
    LE,
    Not,
    Select,
    Solver,
    Store,
    Symbol,
    binary_interpolant,
    is_unsat,
    sequence_interpolant,
)
from pysmt.test import TestCase, main
from pysmt.typing import ArrayType, BVType, INT


HAS_BITWUZLA = "bitwuzla" in get_env().factory.all_solvers()
HAS_CVC5 = "cvc5" in get_env().factory.all_solvers()


class TestBitwuzlaSolver(TestCase):

    @unittest.skipUnless(HAS_BITWUZLA, "Bitwuzla is not available")
    def test_incremental_solving_and_model(self):
        x = Symbol("bitwuzla_x", BVType(8))
        value = BV(10, 8)

        with Solver(name="bitwuzla", logic=QF_BV) as solver:
            solver.add_assertion(Equals(x, value))
            self.assertTrue(solver.solve())
            self.assertEqual(solver.get_value(x), value)

            solver.push()
            solver.add_assertion(Not(Equals(x, value)))
            self.assertFalse(solver.solve())
            solver.pop()

            self.assertTrue(solver.solve())


class TestNativeTermBridge(TestCase):

    def test_arbitrary_precision_numeral(self):
        x = Symbol("huge_integer_x", INT)
        digits = "9" * 5000
        formula = term_to_pysmt(
            "(= huge_integer_x %s)" % digits,
            QF_LIA,
            "(declare-fun huge_integer_x () Int)",
            get_env(),
            [x],
        )

        value = formula.arg(1)
        self.assertTrue(value.is_int_constant())
        self.assertEqual(value.constant_value() % (10 ** 20), (10 ** 20) - 1)
        if HAS_CVC5:
            self.assertTrue(
                is_unsat(
                    And(formula, Not(formula)),
                    solver_name="cvc5",
                    logic=QF_LIA,
                )
            )


class NativeInterpolationMixin(object):

    solver_name = None

    def assert_interpolant(self, a, b, interpolant, logic):
        self.assertIsNotNone(interpolant)
        self.assertTrue(
            is_unsat(
                And(a, Not(interpolant)),
                solver_name=self.solver_name,
                logic=logic,
            )
        )
        self.assertTrue(
            is_unsat(
                And(interpolant, b),
                solver_name=self.solver_name,
                logic=logic,
            )
        )

    def test_binary_and_sequence_bv(self):
        x = Symbol("%s_x" % self.solver_name, BVType(8))
        a = Equals(x, BV(0, 8))
        middle = Equals(x, BV(0, 8))
        b = Equals(x, BV(1, 8))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name=self.solver_name,
            logic=QF_BV,
        )
        self.assert_interpolant(a, b, interpolant, QF_BV)

        sequence = sequence_interpolant(
            [a, middle, b],
            solver_name=self.solver_name,
            logic=QF_BV,
        )
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 2)
        self.assert_interpolant(a, And(middle, b), sequence[0], QF_BV)
        self.assert_interpolant(And(sequence[0], middle), b, sequence[1], QF_BV)

    def test_sat_problem_has_no_interpolant(self):
        x = Symbol("%s_sat_x" % self.solver_name, BVType(8))
        formula = Equals(x, BV(0, 8))
        interpolant = binary_interpolant(
            formula,
            formula,
            solver_name=self.solver_name,
            logic=QF_BV,
        )
        self.assertIsNone(interpolant)

    def test_array_interpolation(self):
        index_type = BVType(8)
        value_type = BVType(8)
        array = Symbol(
            "%s_array" % self.solver_name,
            ArrayType(index_type, value_type),
        )
        index = Symbol("%s_index" % self.solver_name, index_type)
        selected = Select(array, index)
        a = Equals(selected, BV(0, 8))
        b = Equals(selected, BV(1, 8))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name=self.solver_name,
            logic=QF_ABV,
        )
        self.assert_interpolant(a, b, interpolant, QF_ABV)

    def test_constant_array_equality_is_decided(self):
        """Equalities involving constant arrays must not defeat the backend.

        Both backends used to fail on this shape rather than answer it:
        Bitwuzla returned `unknown` ("Equality over constant arrays not fully
        supported yet") and cvc5 raised an internal error about write-chains
        connecting two different constant arrays.  Over a finite index sort the
        equality is expressible elementwise, which is decidable for both.

        The formula is unsatisfiable: four indices, but the left side can
        differ from its 0-default at only two of them and the right side from
        its 1-default at only one, so some index has 0 on the left and 1 on the
        right.
        """
        index_type = BVType(2)
        value_type = BVType(2)
        suffix = self.solver_name
        i = Symbol("%s_ca_i" % suffix, index_type)
        j = Symbol("%s_ca_j" % suffix, index_type)
        k = Symbol("%s_ca_k" % suffix, index_type)
        x = Symbol("%s_ca_x" % suffix, value_type)
        y = Symbol("%s_ca_y" % suffix, value_type)
        z = Symbol("%s_ca_z" % suffix, value_type)

        left = Store(Store(Array(index_type, BV(0, 2)), i, x), j, y)
        right = Store(Array(index_type, BV(1, 2)), k, z)

        self.assertTrue(
            is_unsat(
                Equals(left, right),
                solver_name=self.solver_name,
                logic=QF_ABV,
            )
        )

    def test_quoted_symbol_identity_is_preserved(self):
        state = Symbol("|%s_state|" % self.solver_name, BVType(8))
        a = Equals(state, BV(0, 8))
        b = Equals(state, BV(1, 8))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name=self.solver_name,
            logic=QF_BV,
        )
        self.assert_interpolant(a, b, interpolant, QF_BV)
        self.assertIn(state, interpolant.get_free_variables())


class TestBatchedTermParsing(TestCase):
    """Parsing several native terms at once must match parsing them one by one.

    The declaration block is re-read for every term when they are parsed
    separately, which made the interpolant round trip quadratic in the bound.
    Batching is only safe if it is indistinguishable from the singular path,
    including the order of the results.
    """

    def test_batch_matches_one_by_one(self):
        x = Symbol("batch_x", INT)
        y = Symbol("|batch quoted y|", INT)
        declarations = "\n".join(
            "(declare-fun %s () Int)" % name for name in ("batch_x", "|batch quoted y|")
        )
        terms = ["(= batch_x 1)", "(< batch_x |batch quoted y|)", "(= |batch quoted y| 7)"]

        batched = terms_to_pysmt(terms, QF_LIA, declarations, get_env(), [x, y])
        one_by_one = [
            term_to_pysmt(term, QF_LIA, declarations, get_env(), [x, y])
            for term in terms
        ]

        self.assertEqual(batched, one_by_one)
        self.assertEqual(len(batched), 3)
        # The quoted name survives the round trip rather than being split.
        self.assertIn(y, batched[2].get_free_variables())

    def test_empty_batch(self):
        self.assertEqual(terms_to_pysmt([], QF_LIA, "", get_env()), [])


class TestArrayEqualityExpansion(TestCase):
    """The rewrite that keeps constant arrays out of the native backends.

    These need no solver: they pin the two properties the backends rely on --
    that an array-free formula is returned untouched, and that nothing
    array-sorted survives in one that is rewritten.
    """

    def test_array_free_formula_is_unchanged(self):
        x = Symbol("expand_x", BVType(8))
        formula = And(Equals(x, BV(0, 8)), Not(Equals(x, BV(1, 8))))
        self.assertEqual(
            expand_array_equalities(formula, get_env()),
            formula,
        )

    def test_array_free_formula_is_not_walked(self):
        """QF_BV must not pay for a rewrite that cannot change anything.

        The interpolation shortcuts build a fresh expander per call, so a walk
        here is a full DAG rebuild of the unrolling on every interpolant --
        measurably slower on exactly the large BV tasks that sit near the time
        limit. The environment's theory oracle answers instead, and it
        memoizes for the whole process.
        """
        x = Symbol("nowalk_x", BVType(8))
        formula = And(Equals(x, BV(0, 8)), Not(Equals(x, BV(1, 8))))

        expander = ArrayEqualityExpander(env=get_env())
        walked = []
        original_walk = expander.walk
        expander.walk = lambda f, **kw: (walked.append(f),
                                         original_walk(f, **kw))[1]

        self.assertEqual(
            expand_array_equalities(formula, get_env(), expander),
            formula,
        )
        self.assertEqual(walked, [])

    def test_constant_array_equality_loses_its_array_terms(self):
        index_type = BVType(2)
        array = Symbol("expand_a", ArrayType(index_type, BVType(2)))
        formula = Equals(array, Array(index_type, BV(0, 2)))

        expanded = expand_array_equalities(formula, get_env())

        # The equality becomes one conjunct per index, and each read of the
        # constant array folds to the constant itself, so no array-sorted
        # subterm is left for the backend to choke on.
        stc = get_env().stc
        for node in expanded.get_atoms():
            for sub in node.args():
                self.assertFalse(stc.get_type(sub).is_array_type())
        self.assertEqual(expanded.get_free_variables(), {array})

    def test_wide_index_sort_is_left_alone(self):
        """Refusing to expand keeps a wide index sort from blowing up."""
        index_type = BVType(32)
        array = Symbol("expand_wide", ArrayType(index_type, BVType(2)))
        formula = Equals(array, Array(index_type, BV(0, 2)))
        self.assertEqual(
            expand_array_equalities(formula, get_env()),
            formula,
        )


@unittest.skipUnless(HAS_BITWUZLA, "Bitwuzla is not available")
class TestBitwuzlaInterpolation(NativeInterpolationMixin, TestCase):
    solver_name = "bitwuzla"

    def test_constant_array_interpolation(self):
        """Interpolate across MoXI's array-initialization idiom.

        A state variable initialized to a constant array, stepped by a store,
        and read back in the query.  Bitwuzla answered `unknown` here before
        the equality was expanded -- and, once it could decide the formula,
        crashed inside `get_interpolant` while an array term remained -- so
        this covers both halves of that fix.

        Bitwuzla only: cvc5's interpolator rejects the constant-array logic
        (QF_ABV*) up front, independently of this rewrite.
        """
        index_type = BVType(2)
        value_type = BVType(2)
        state = Symbol("bzla_ca_state", ArrayType(index_type, value_type))
        step = Symbol("bzla_ca_step", ArrayType(index_type, value_type))

        a = And(
            Equals(state, Array(index_type, BV(0, 2))),
            Equals(step, Store(state, BV(1, 2), BV(3, 2))),
        )
        b = Not(Equals(Select(step, BV(1, 2)), BV(3, 2)))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name=self.solver_name,
            logic=QF_ABV,
        )
        self.assert_interpolant(a, b, interpolant, QF_ABV)


@unittest.skipUnless(HAS_CVC5, "cvc5 is not available")
class TestCVC5Interpolation(NativeInterpolationMixin, TestCase):
    solver_name = "cvc5"

    def test_solver_exit_releases_native_context(self):
        x = Symbol("cvc5_lifecycle_x", INT)
        solver = Solver(name="cvc5", logic=QF_LIA)
        solver.add_assertion(Equals(x, Int(1)))
        self.assertTrue(solver.solve())

        solver_ref = weakref.ref(solver)
        converter_ref = weakref.ref(solver.converter)
        solver.exit()

        self.assertTrue(solver._destroyed)
        self.assertFalse(hasattr(solver, "converter"))
        self.assertFalse(hasattr(solver, "cvc5_solver"))

        del solver
        gc.collect()
        self.assertIsNone(converter_ref())
        self.assertIsNone(solver_ref())

    def test_integer_interpolation(self):
        x = Symbol("cvc5_lia_x", INT)
        y = Symbol("cvc5_lia_y", INT)
        z = Symbol("cvc5_lia_z", INT)
        a = And(LE(x, Int(0)), LE(y, x))
        b = And(GE(y, z), Equals(z, Int(1)))

        interpolant = binary_interpolant(
            a,
            b,
            solver_name=self.solver_name,
            logic=QF_LIA,
        )
        self.assert_interpolant(a, b, interpolant, QF_LIA)


if __name__ == "__main__":
    main()
