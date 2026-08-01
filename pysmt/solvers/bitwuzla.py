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
"""Native Bitwuzla solver and interpolation support."""

from typing import Iterable, Optional, Sequence

from pysmt.decorators import catch_conversion_error, clear_pending_pop
from pysmt.environment import Environment
from pysmt.exceptions import (
    ConvertExpressionError,
    InternalSolverError,
    PysmtValueError,
    SolverAPINotFound,
    SolverReturnedUnknownResultError,
)
from pysmt.fnode import FNode
from pysmt.logics import (
    ARRAYS_CONST_LOGICS,
    QF_ABV,
    QF_AUFBV,
    QF_BV,
    QF_UFBV,
)
from pysmt.oracles import get_logic
from pysmt.solvers._array_rewriter import (
    expand_array_equalities,
    make_array_expander,
)
from pysmt.solvers._smtlib_bridge import (
    collect_symbols,
    declaration_script,
    make_renamer,
    make_safe_symbol_map,
    node_to_smtlib,
    normalize_logic_name,
    repair_quoted_symbols,
    rename_symbols,
    symbol_declaration,
    term_to_pysmt,
    terms_to_pysmt,
)
from pysmt.solvers.eager import EagerModel
from pysmt.solvers.interpolation import Interpolator
from pysmt.solvers.smtlib import SmtLibBasicSolver
from pysmt.solvers.solver import Converter, IncrementalTrackingSolver, SolverOptions

try:
    from bitwuzla import (
        BitwuzlaException,
        Option,
        Options,
        Parser,
        Result,
        TermManager,
    )
except ImportError:
    raise SolverAPINotFound


BITWUZLA_LOGICS = [
    QF_BV,
    QF_UFBV,
    QF_ABV,
    QF_AUFBV,
] + list(
    filter(
        lambda logic: logic.name in {"QF_ABV*", "QF_AUFBV*"},
        ARRAYS_CONST_LOGICS,
    )
)


class BitwuzlaOptions(SolverOptions):
    """Options for the native Bitwuzla backend."""

    def __init__(self, **base_options):
        SolverOptions.__init__(self, **base_options)
        if self.unsat_cores_mode is not None:
            raise PysmtValueError(
                "'unsat_cores_mode' is not supported by the Bitwuzla backend."
            )

    def __call__(self, solver):
        solver._native_options.set(Option.PRODUCE_MODELS, self.generate_models)

        if self.random_seed is not None:
            solver._native_options.set(Option.SEED, self.random_seed)

        for name, value in self.solver_options.items():
            option_name = str(name).upper().replace("-", "_").replace(".", "_")
            try:
                option = Option.__members__[option_name]
            except KeyError:
                raise PysmtValueError(
                    "Unknown Bitwuzla option '%s'. Accepted names are: %s"
                    % (name, ", ".join(sorted(Option.__members__)))
                )
            try:
                solver._native_options.set(option, value)
            except (RuntimeError, TypeError, ValueError) as ex:
                raise PysmtValueError(
                    "Error setting Bitwuzla option '%s=%s': %s"
                    % (name, value, ex)
                )


class BitwuzlaConverter(Converter):
    """SMT-LIB-backed converter for Bitwuzla's native parser."""

    def __init__(self, solver):
        self.solver = solver

    @catch_conversion_error
    def convert(self, formula):
        try:
            return self.solver._convert(formula)
        except BitwuzlaException as ex:
            raise ConvertExpressionError(
                message="Bitwuzla could not convert the expression: %s" % ex,
                expression=formula,
            )

    def back(self, term, *args, **kwargs):
        return self.solver._back(term)


class BitwuzlaSolver(IncrementalTrackingSolver, SmtLibBasicSolver):
    """Incremental pySMT solver backed by Bitwuzla."""

    LOGICS = BITWUZLA_LOGICS
    OptionsClass = BitwuzlaOptions

    def __init__(self, environment: Environment, logic, **options):
        IncrementalTrackingSolver.__init__(
            self,
            environment=environment,
            logic=logic,
            **options
        )
        self._new_context()
        self.converter = BitwuzlaConverter(self)

    def _new_context(self):
        self._term_manager = TermManager()
        self._native_options = Options()
        self.options(self)
        self._parser = Parser(self._term_manager, self._native_options)
        self._parser.parse(
            "(set-logic %s)" % normalize_logic_name(self.logic),
            True,
            False,
        )
        self.bitwuzla = self._parser.bitwuzla()
        self._declared_symbols = set()
        self._declaration_lines = []
        self._symbol_map = {}
        # One renamer per context, sharing the (growing) symbol map: the
        # assertion stack of an incremental solve repeats most of its subterms,
        # and a per-call walker rebuilt all of them each time.
        self._renamer = make_renamer(self._symbol_map, self.environment)
        self._array_expander = make_array_expander(self.environment)

    def _ensure_declarations(self, formulas: Iterable[FNode]):
        new_symbols = [
            symbol
            for symbol in collect_symbols(formulas)
            if symbol not in self._declared_symbols
        ]
        if not new_symbols:
            return

        self._symbol_map.update(
            make_safe_symbol_map(
                formulas,
                self.environment,
                "bitwuzla",
                symbols=new_symbols,
            )
        )
        native_symbols = [self._symbol_map[symbol] for symbol in new_symbols]
        declarations = "\n".join(
            symbol_declaration(symbol) for symbol in native_symbols
        )
        self._parser.parse(repair_quoted_symbols(declarations), True, False)
        self._declared_symbols.update(new_symbols)
        self._declaration_lines.extend(
            symbol_declaration(symbol) for symbol in native_symbols
        )

    def _convert(self, formula: FNode):
        # Bitwuzla is incomplete on equalities involving constant arrays and
        # answers `unknown` for them; stating those equalities elementwise is
        # an equivalence over a finite index sort and keeps the query decidable.
        formula = expand_array_equalities(
            formula,
            self.environment,
            self._array_expander,
        )
        self._ensure_declarations([formula])
        native_formula = rename_symbols(
            formula,
            self._symbol_map,
            self.environment,
            self._renamer,
        )
        return self._parser.parse_term(node_to_smtlib(native_formula))

    def _back(self, term):
        return term_to_pysmt(
            term,
            self.logic,
            "\n".join(self._declaration_lines),
            self.environment,
            self._declared_symbols,
            self._symbol_map,
        )

    @clear_pending_pop
    def _reset_assertions(self):
        self._new_context()

    @clear_pending_pop
    def _add_assertion(self, formula: FNode, named: Optional[str] = None):
        self._assert_is_boolean(formula)
        if named is not None:
            raise PysmtValueError(
                "Named assertions are not supported by the Bitwuzla backend."
            )
        self.bitwuzla.assert_formula(self.converter.convert(formula))
        return formula

    @clear_pending_pop
    def _solve(self, assumptions=None):
        native_assumptions = []
        if assumptions is not None:
            native_assumptions = [
                self.converter.convert(assumption) for assumption in assumptions
            ]

        try:
            result = self.bitwuzla.check_sat(*native_assumptions)
        except RuntimeError as ex:
            raise InternalSolverError(str(ex))

        if result == Result.SAT:
            return True
        if result == Result.UNSAT:
            return False
        raise SolverReturnedUnknownResultError()

    @clear_pending_pop
    def _push(self, levels=1):
        self.bitwuzla.push(levels)

    @clear_pending_pop
    def _pop(self, levels=1):
        self.bitwuzla.pop(levels)

    def get_value(self, item):
        self._assert_no_function_type(item)
        try:
            value = self.bitwuzla.get_value(self.converter.convert(item))
        except RuntimeError as ex:
            raise InternalSolverError(str(ex))
        return self.converter.back(value)

    def get_model(self):
        assignment = {}
        for symbol in self._declared_symbols:
            if not symbol.symbol_type().is_function_type():
                assignment[symbol] = self.get_value(symbol)
        return EagerModel(assignment=assignment, environment=self.environment)

    def declare_variable(self, var):
        self._ensure_declarations([var])

    def _exit(self):
        del self.bitwuzla
        del self._parser
        del self._term_manager


class BitwuzlaInterpolator(Interpolator):
    """Craig and sequence interpolation backed by Bitwuzla."""

    LOGICS = BITWUZLA_LOGICS

    def __init__(self, environment: Environment, logic=None):
        Interpolator.__init__(self)
        self.environment = environment
        self.logic = logic

    def _check_logic(self, formulas: Sequence[FNode]):
        for formula in formulas:
            detected = get_logic(formula, self.environment)
            if not any(detected <= supported for supported in self.LOGICS):
                raise PysmtValueError(
                    "Logic not supported by Bitwuzla interpolation. "
                    "(detected logic is: %s)" % detected
                )

    def _context(self, formulas: Sequence[FNode]):
        manager = TermManager()
        options = Options()
        options.set(Option.PRODUCE_INTERPOLANTS, True)
        options.set(Option.INTERPOLANTS_SIMP, True)
        parser = Parser(manager, options)
        parser.parse(
            "(set-logic %s)" % normalize_logic_name(self.logic),
            True,
            False,
        )
        # Equalities between constant arrays make Bitwuzla give up with
        # `unknown` before it ever reaches interpolation. Stating them
        # elementwise is an equivalence and introduces no new symbols, so the
        # interpolant comes back already expressed in the original signature.
        expander = make_array_expander(self.environment)
        formulas = [
            expand_array_equalities(formula, self.environment, expander)
            for formula in formulas
        ]
        # The blocks of a sequence interpolation overlap heavily, so the
        # symbols are collected once and the renamer is shared across them.
        symbols = collect_symbols(formulas)
        symbol_map = make_safe_symbol_map(
            formulas,
            self.environment,
            "bitwuzla_itp",
            symbols=symbols,
        )
        renamer = make_renamer(symbol_map, self.environment)
        native_formulas = [
            rename_symbols(formula, symbol_map, self.environment, renamer)
            for formula in formulas
        ]
        # Renaming is injective and total over the collected symbols, so this
        # is exactly what collecting from the renamed formulas would yield;
        # sorted the same way to keep the emitted script unchanged.
        native_symbols = sorted(
            (symbol_map[symbol] for symbol in symbols),
            key=lambda symbol: symbol.symbol_name(),
        )
        declarations, symbols = declaration_script(
            native_formulas,
            symbols=native_symbols,
        )
        if declarations:
            parser.parse(repair_quoted_symbols(declarations), True, False)
        return (
            parser,
            parser.bitwuzla(),
            declarations,
            symbols,
            symbol_map,
            native_formulas,
        )

    def binary_interpolant(self, a: FNode, b: FNode):
        self._check_logic([a, b])
        (parser, solver, declarations, symbols,
         symbol_map, native_formulas) = self._context([a, b])
        a_term = parser.parse_term(node_to_smtlib(native_formulas[0]))
        b_term = parser.parse_term(node_to_smtlib(native_formulas[1]))
        solver.assert_formula(a_term, b_term)

        result = solver.check_sat()
        if result == Result.SAT:
            return None
        if result == Result.UNKNOWN:
            raise SolverReturnedUnknownResultError()

        interpolant = solver.get_interpolant([a_term])
        return term_to_pysmt(
            interpolant,
            self.logic,
            declarations,
            self.environment,
            symbols,
            symbol_map,
        )

    def sequence_interpolant(self, formulas: Sequence[FNode]):
        formulas = list(formulas)
        if len(formulas) < 2:
            raise PysmtValueError("interpolation needs at least 2 formulae")
        self._check_logic(formulas)

        (parser, solver, declarations, symbols,
         symbol_map, native_formulas) = self._context(formulas)
        native_formulas = [
            parser.parse_term(node_to_smtlib(formula))
            for formula in native_formulas
        ]
        solver.assert_formula(*native_formulas)

        result = solver.check_sat()
        if result == Result.SAT:
            return None
        if result == Result.UNKNOWN:
            raise SolverReturnedUnknownResultError()

        native_interpolants = solver.get_interpolants(
            [[formula] for formula in native_formulas[:-1]]
        )
        return terms_to_pysmt(
            native_interpolants,
            self.logic,
            declarations,
            self.environment,
            symbols,
            symbol_map,
        )

    def _exit(self):
        pass
