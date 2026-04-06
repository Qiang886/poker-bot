"""Tests for EV-driven decision making in PostflopEngine."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.hand_analysis import MadeHandType
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range


def make_engine():
    return PostflopEngine()


def decide(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    street="flop",
    villain_profile=None,
    is_multiway=False,
):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.UTG), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    return engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=Position.BTN,
        street=street,
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=is_multiway,
    )


# ---------------------------------------------------------------------------
# EV helper method tests
# ---------------------------------------------------------------------------

def test_calculate_bet_ev_positive():
    """EV(bet) should be positive when equity and fold_equity are high."""
    engine = make_engine()
    ev = engine._calculate_bet_ev(
        equity=0.70, pot=100.0, bet_size=60.0, fold_equity=0.50
    )
    # fold_eq * pot + (1 - fold_eq) * (equity * (pot + 2*bet) - bet)
    # = 0.5 * 100 + 0.5 * (0.7 * 220 - 60) = 50 + 0.5*(154 - 60) = 50 + 47 = 97
    assert ev > 0


def test_calculate_check_ev_proportional_to_equity():
    """EV(check) should equal equity * pot."""
    engine = make_engine()
    ev = engine._calculate_check_ev(equity=0.60, pot=100.0)
    assert abs(ev - 60.0) < 0.01


def test_calculate_raise_ev_positive():
    """EV(raise) should be positive for strong hands with fold equity."""
    engine = make_engine()
    ev = engine._calculate_raise_ev(
        equity=0.80, pot=100.0, to_call=30.0, raise_size=90.0, fold_equity=0.55
    )
    assert ev > 0


def test_bet_ev_higher_than_check_leads_to_bet():
    """When EV(bet) >> EV(check), decision should be bet."""
    # Strong hand on dry board → EV(bet) > EV(check)
    d = decide("AhKs", "As2d7h", pot=10.0, to_call=0.0, street="flop")
    assert d.action in ("bet", "all-in")


def test_weak_hand_ev_leads_to_check():
    """Air hand (no pair, no draw) should prefer check over bet."""
    # 2h3h on AsTc9d — no pair, no draw for 2h3h
    d = decide("2h3h", "AsTc9d", pot=10.0, to_call=0.0, street="flop")
    assert d.action == "check"


def test_ev_call_positive_leads_to_call():
    """When EV(call) > 0, bot should call (positive expected value)."""
    # Strong hand with equity well above pot odds should call
    d = decide("AhKc", "AsQd7h", pot=10.0, to_call=3.0, street="flop")
    assert d.action in ("call", "raise", "all-in")


def test_ev_fold_when_equity_below_pot_odds():
    """When pot odds require more equity than hero has, fold."""
    # Weak hand facing a large bet
    d = decide("2h3c", "AsTcKd", pot=10.0, to_call=9.0, street="flop")
    assert d.action == "fold"


def test_monster_hand_sanity_override():
    """Monster hand (TRIPS+) should call even if theoretical EV is marginal."""
    # Set of 7s — should never fold
    d = decide("7h7d", "7s2cKd5h", pot=10.0, to_call=8.0, street="flop")
    assert d.action in ("call", "raise", "all-in")


def test_air_sanity_override_no_bet():
    """Air hand (no pair, no draw) should not bet even if EV(bet) looks ok."""
    # 2h3c on AsKdQh — pure air, no draw (no flush draw, no straight draw)
    d = decide("2h3c", "AsKdQh", pot=10.0, to_call=0.0, street="flop")
    assert d.action == "check"


def test_get_fold_equity_default():
    """Without villain_profile, fold_equity should default to 0.45."""
    engine = make_engine()
    fe = engine._get_fold_equity(None, "flop")
    assert abs(fe - 0.45) < 0.01


def test_get_fold_equity_from_profile():
    """Fold equity should be read from villain profile stats."""
    from src.opponent import VillainProfile
    vp = VillainProfile()
    vp.stats.fold_to_flop_cbet = 0.65
    vp.stats.fold_to_turn_cbet = 0.55
    vp.stats.fold_to_river_cbet = 0.40
    engine = make_engine()
    assert abs(engine._get_fold_equity(vp, "flop") - 0.65) < 0.01
    assert abs(engine._get_fold_equity(vp, "turn") - 0.55) < 0.01
    assert abs(engine._get_fold_equity(vp, "river") - 0.40) < 0.01
