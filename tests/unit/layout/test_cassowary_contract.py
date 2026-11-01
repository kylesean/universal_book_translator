from __future__ import annotations

import pytest

kiwisolver = pytest.importorskip("kiwisolver")

pytestmark = pytest.mark.fast


def test_solver_honours_strength_ordering() -> None:
    solver = kiwisolver.Solver()
    y = kiwisolver.Variable("y")
    gap = kiwisolver.Variable("gap")

    solver.addConstraint(y + gap >= 12)
    solver.addConstraint((y >= 100) | kiwisolver.strength.strong)
    solver.addConstraint((gap == 6) | kiwisolver.strength.medium)
    solver.updateVariables()

    assert y.value() == pytest.approx(100.0), "the strong preference wins over the default"
    assert gap.value() == pytest.approx(6.0)
    assert y.value() + gap.value() >= 12.0


def test_line_count_is_an_input_not_a_solved_variable() -> None:
    solver = kiwisolver.Solver()
    line_count = 7
    leading = kiwisolver.Variable("leading")
    height = kiwisolver.Variable("height")

    solver.addConstraint(height == (line_count - 1) * leading)
    solver.addConstraint((leading >= 0) | kiwisolver.strength.required)
    solver.addConstraint((height <= 120) | kiwisolver.strength.strong)
    solver.updateVariables()

    assert height.value() == pytest.approx((line_count - 1) * leading.value())
    assert height.value() <= 120.0 + 1e-9
    assert line_count == 7, "the solver cannot change a measured count"


def test_an_unsatisfiable_system_raises_instead_of_guessing() -> None:
    solver = kiwisolver.Solver()
    y = kiwisolver.Variable("y")

    with pytest.raises(kiwisolver.UnsatisfiableConstraint):
        solver.addConstraint(y >= 10)
        solver.addConstraint(y <= 5)
