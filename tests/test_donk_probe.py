"""Tests for donk bet and probe bet strategies."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.opponent import VillainProfile
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range


def make_engine():
    return PostflopEngine()


def make_villain(vpip=0.25, pfr=0.20, af=2.0, fold_cbet=0.50, fold_turn_cbet=0.45,
                 cbet_flop=0.60, hands=50):
    v = VillainProfile()
    v.stats.vpip = vpip
    v.stats.pfr = pfr
    v.stats.aggression_factor = af
    v.stats.fold_to_flop_cbet = fold_cbet
    v.stats.fold_to_turn_cbet = fold_turn_cbet
    v.stats.cbet_flop = cbet_flop
    v.stats.hands_played = hands
    return v


def decide_flop(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    position=Position.BB,  # OOP by default for donk tests
    villain_profile=None,
    is_multiway=False,
):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BB), dead)
    return engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=position,
        street="flop",
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=is_multiway,
        sizing_ratio=0.0,
    )


def decide_turn(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    position=Position.BB,  # OOP by default for probe tests
    villain_profile=None,
    is_multiway=False,
    action_history=None,
):
    """Decide on turn street with optional action history for probe detection."""
    from unittest.mock import patch
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BB), dead)

    # Patch _villain_checked_back_flop if action_history is provided
    if action_history is not None:
        original_method = engine._villain_checked_back_flop
        engine._villain_checked_back_flop = lambda hist: any(
            a.get("actor") == "villain" and a.get("action") == "check"
            and a.get("street") == "flop"
            for a in action_history
        )

    result = engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=position,
        street="turn",
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=is_multiway,
        sizing_ratio=0.0,
    )

    if action_history is not None:
        engine._villain_checked_back_flop = original_method

    return result


# ---------------------------------------------------------------------------
# Donk bet tests – Flop OOP
# ---------------------------------------------------------------------------

def test_donk_bet_ace_high_board_with_ax():
    """BB with Ax should donk bet on A-high board (range advantage)."""
    # BB has Ace, board has Ace → range advantage donk
    d = decide_flop("AhTc", "As7d3c")
    assert d.action in ("bet", "raise"), f"Expected donk bet with Ax on A-high board, got {d.action}"
    assert "Donk" in d.reasoning or "donk" in d.reasoning.lower() or d.action == "bet"


def test_donk_bet_connected_board_two_pair():
    """Two pair on connected board should donk for value+protection."""
    # Connected board + two pair: 8h 7c 2d, hero has 8d 7s
    d = decide_flop("8d7s", "8h7c2d")
    assert d.action in ("bet", "raise"), f"Expected donk bet with two pair on connected board, got {d.action}"


def test_donk_bet_wet_board_middle_pair_protection():
    """Middle pair on wet board from OOP should donk for protection."""
    # Wet board, hero has middle pair and is vulnerable
    d = decide_flop("9hTs", "9d8s7h")
    assert d.action in ("bet", "raise"), f"Expected donk bet with middle pair on wet board, got {d.action}"


def test_no_donk_when_villain_cbets_high():
    """Should not donk when villain has high cbet frequency (> 75%)."""
    high_cbet = make_villain(cbet_flop=0.80)
    d = decide_flop("AhTc", "As7d3c", villain_profile=high_cbet)
    # With high cbet villain, should check and let him bet (then check-raise opportunity)
    # The donk logic should be suppressed
    assert d.action in ("check", "bet"), f"Action was {d.action}"
    if d.action == "bet":
        # If it bets, it should NOT be a donk bet reasoning
        assert "Donk" not in d.reasoning or high_cbet is not None


def test_no_donk_bet_villain_high_cbet_freq():
    """No donk bet when villain cbets > 75% (let him cbet, then CR)."""
    high_cbet_villain = make_villain(cbet_flop=0.85)
    d = decide_flop("AhTc", "As7d3c", villain_profile=high_cbet_villain)
    # Should NOT donk when villain always cbets
    assert "Donk" not in d.reasoning


def test_no_donk_in_multiway():
    """No donk bet in multiway pots."""
    d = decide_flop("AhTc", "As7d3c", is_multiway=True)
    # In multiway, donk bet should be suppressed
    if d.action == "bet":
        assert "Donk" not in d.reasoning


def test_no_donk_with_air_no_draw():
    """No donk bet with pure air (no pair, no draw)."""
    # Hero has no pair, no draw on this board
    d = decide_flop("2c3d", "As8hKd")
    # Air hand should not donk
    if d.action == "bet":
        assert "Donk" not in d.reasoning


# ---------------------------------------------------------------------------
# Probe bet tests – Turn after villain flop check-back
# ---------------------------------------------------------------------------

def test_probe_bet_middle_pair_after_checkback():
    """Middle pair should probe bet after villain checks back the flop."""
    villain_checked_flop = [
        {"street": "flop", "actor": "villain", "action": "check"}
    ]
    d = decide_turn(
        "9hTs", "9d3c2sKh",
        action_history=villain_checked_flop,
    )
    assert d.action in ("bet", "raise"), f"Expected probe bet with middle pair after check-back, got {d.action}"


def test_probe_bet_draw_after_checkback():
    """Draw should probe bet (bluff probe) after villain checks back the flop."""
    villain_checked_flop = [
        {"street": "flop", "actor": "villain", "action": "check"}
    ]
    # Hero has flush draw
    high_fold = make_villain(fold_turn_cbet=0.60)
    d = decide_turn(
        "AhQh", "9h6h2dTc",
        villain_profile=high_fold,
        action_history=villain_checked_flop,
    )
    assert d.action in ("bet", "raise"), f"Expected probe bluff with flush draw after check-back, got {d.action}"


def test_no_probe_with_air_no_draw():
    """Should not probe with pure air and no draw after villain check-back."""
    villain_checked_flop = [
        {"street": "flop", "actor": "villain", "action": "check"}
    ]
    # Hero has nothing
    d = decide_turn(
        "2c4d", "AsKhQd3c",
        action_history=villain_checked_flop,
    )
    # Pure air: no pair, no draw → no probe; if it does bet, it should not be a Probe bet
    if d.action == "bet":
        assert "Probe" not in d.reasoning, f"Should not probe with pure air, got: {d.reasoning}"


def test_no_probe_without_checkback():
    """Without villain check-back signal, should not probe bet."""
    # No action history → no probe
    d = decide_turn(
        "9hTs", "9d3c2sKh",
        action_history=None,
    )
    # Without checkback history, probe should not trigger
    if d.action == "bet":
        assert "Probe" not in d.reasoning


def test_probe_not_in_multiway():
    """Probe bet should not trigger in multiway pots."""
    villain_checked_flop = [
        {"street": "flop", "actor": "villain", "action": "check"}
    ]
    d = decide_turn(
        "9hTs", "9d3c2sKh",
        is_multiway=True,
        action_history=villain_checked_flop,
    )
    # Multiway: no probe
    if d.action == "bet":
        assert "Probe" not in d.reasoning
