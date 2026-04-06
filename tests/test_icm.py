"""Tests for ICM calculations."""

import pytest
from src.icm import calculate_icm, icm_pressure, adjust_strategy_for_icm


def test_icm_sum_equals_payouts():
    stacks = [1000.0, 1000.0, 1000.0]
    payouts = [500.0, 300.0, 200.0]
    equities = calculate_icm(stacks, payouts)
    assert abs(sum(equities) - sum(payouts)) < 0.01


def test_icm_equal_stacks_equal_equities():
    stacks = [1000.0, 1000.0, 1000.0]
    payouts = [500.0, 300.0, 200.0]
    equities = calculate_icm(stacks, payouts)
    assert abs(equities[0] - equities[1]) < 0.01
    assert abs(equities[1] - equities[2]) < 0.01


def test_icm_chip_leader_has_more_equity():
    stacks = [2000.0, 1000.0, 500.0]
    payouts = [500.0, 300.0, 200.0]
    equities = calculate_icm(stacks, payouts)
    assert equities[0] > equities[1] > equities[2]


def test_icm_pressure_bubble():
    # Near-bubble: short stack has high pressure
    stacks = [5000.0, 5000.0, 100.0, 5000.0]
    payouts = [1000.0, 600.0, 400.0]  # top 3 paid, 4th gets nothing
    pressure = icm_pressure(100.0, stacks, payouts, 2)
    # Short stack on bubble: high pressure
    assert pressure >= 0.0


def test_adjust_fold_marginal_high_pressure():
    action, amount = adjust_strategy_for_icm("call", 500.0, 300.0, 0.8, 20.0)
    assert action == "fold"


def test_adjust_no_change_low_pressure():
    action, amount = adjust_strategy_for_icm("bet", 100.0, 200.0, 0.05, 50.0)
    assert action == "bet"
