"""Tests for Turn independent decision framework."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.opponent import VillainProfile
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range


def make_engine():
    return PostflopEngine()


def make_villain(vpip, pfr, af=2.0, fold_river=0.45, fold_cbet=0.50, hands=50,
                 fold_turn_cbet=0.45):
    v = VillainProfile()
    v.stats.vpip = vpip
    v.stats.pfr = pfr
    v.stats.aggression_factor = af
    v.stats.fold_to_river_cbet = fold_river
    v.stats.fold_to_flop_cbet = fold_cbet
    v.stats.fold_to_turn_cbet = fold_turn_cbet
    v.stats.hands_played = hands
    return v


def decide(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    position=Position.BTN,
    villain_profile=None,
    is_multiway=False,
    sizing_ratio=0.0,
    barrel_plan=None,
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
        position=position,
        street="turn",
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=barrel_plan,
        is_multiway=is_multiway,
        sizing_ratio=sizing_ratio,
    )


# ---------------------------------------------------------------------------
# Turn blank → continue barrel plan / value bet
# ---------------------------------------------------------------------------

def test_turn_blank_strong_hand_bets():
    """Blank turn card with strong hand should continue betting."""
    # Flop: Ah Kd 7c, Turn: 2s (blank)
    d = decide("AhKc", "AhKd7c2s", pot=10.0)
    assert d.action in ("bet", "raise", "all-in"), f"Expected bet on blank turn, got {d.action}"


def test_turn_blank_value_bet_tptk():
    """TPTK on blank turn should value bet."""
    # Flop: As 8d 3h, Turn: 2c (blank)
    d = decide("AhKd", "As8d3h2c", pot=12.0)
    assert d.action in ("bet", "raise", "all-in"), f"Expected bet with TPTK on blank turn, got {d.action}"


# ---------------------------------------------------------------------------
# Turn overcard (A or K) → slow down
# ---------------------------------------------------------------------------

def test_turn_scare_card_slows_down_weak_hand():
    """Scare card on turn (A arriving on low board) should slow down weak hand."""
    # Flop: 8s 6d 3h (hero has middle pair), Turn: Ac (scare card)
    d = decide("8h7s", "8s6d3hAc", pot=10.0)
    assert d.action == "check", f"Expected check on scare card turn with middle pair, got {d.action}"


def test_turn_king_overcard_weak_hand_checks():
    """King arriving on low board should make bot cautious with weak hand."""
    # Hero has bottom pair, K arrives
    d = decide("3h2s", "3d5c8hKs", pot=10.0)
    assert d.action == "check", f"Expected check on K overcard with weak hand, got {d.action}"


def test_turn_scare_card_strong_hand_still_bets():
    """Strong hand (TPTK or better) should still bet even with scare card."""
    # Hero has top two pair, K arrives → still value bet
    d = decide("KhKd", "Ks8d3hAc", pot=10.0)
    assert d.action in ("bet", "raise", "all-in"), f"Expected bet with monster on scare card turn, got {d.action}"


# ---------------------------------------------------------------------------
# Turn flush complete → stop bluffing unless hero has flush
# ---------------------------------------------------------------------------

def test_turn_flush_complete_hero_no_flush_stops_bluff():
    """When flush completes on turn and hero doesn't have it, stop bluffing."""
    # Flop: 9h 6h 2d (two hearts), Turn: 7h (flush completes) – hero has no hearts
    d = decide("AcKd", "9h6h2d7h", pot=10.0)
    assert d.action == "check", f"Expected check when flush completes and hero has no flush, got {d.action}"


def test_turn_flush_complete_hero_has_flush_bets():
    """When flush completes and hero has the flush, should bet for value."""
    # Hero has Ah Th, board completes flush
    d = decide("AhTh", "9h6h2d7h", pot=10.0)
    assert d.action in ("bet", "raise", "all-in"), f"Expected bet when hero has the flush, got {d.action}"


# ---------------------------------------------------------------------------
# Turn board pairs → slow down
# ---------------------------------------------------------------------------

def test_turn_board_pairs_weak_hand_slows():
    """Weak hand (no pair from hero cards) should slow down when board pairs on turn."""
    # Flop: 8d 6s 3h, Turn: 8c (pairs board) – hero has no pair, uses board pair
    d = decide("KhQd", "8d6s3h8c", pot=10.0)
    assert d.action == "check", f"Expected check when board pairs and hero has no pair, got {d.action}"


def test_turn_board_pairs_monster_still_bets():
    """Monster (full house/set) should still bet when board pairs."""
    # Hero has 8s8c (pocket 8s), board has 8d so hero makes set, then 8h pairs board giving quads/full house
    d = decide("AhAs", "Ah8d3cAc", pot=10.0)  # hero has two pair, A pairs board
    assert d.action in ("bet", "raise", "all-in"), f"Expected bet with monster when board pairs, got {d.action}"


# ---------------------------------------------------------------------------
# Turn facing double barrel → need stronger hand
# ---------------------------------------------------------------------------

def test_turn_facing_bet_monster_raises():
    """Monster hand should raise facing turn bet."""
    d = decide("7h7d", "7s2cKd5h", pot=10.0, to_call=5.0, sizing_ratio=0.5)
    assert d.action in ("raise", "all-in"), f"Expected raise with set facing turn bet, got {d.action}"


def test_turn_facing_bet_tptk_calls():
    """TPTK should call turn bet with positive EV."""
    d = decide("AhKd", "As9d3h7c", pot=10.0, to_call=5.0, sizing_ratio=0.5)
    assert d.action in ("call", "raise", "all-in"), f"Expected call/raise with TPTK facing turn bet, got {d.action}"


def test_turn_facing_bet_weak_hand_folds():
    """Weak hand (no pair, no draw) should fold to a large turn bet."""
    # No pair, no draw facing a 70% pot bet
    d = decide("Kh8d", "As3c7s4c", pot=10.0, to_call=7.0, sizing_ratio=0.7)
    assert d.action == "fold", f"Expected fold with no-pair/no-draw hand facing large turn bet, got {d.action}"


# ---------------------------------------------------------------------------
# Turn facing overbet → polarized handling
# ---------------------------------------------------------------------------

def test_turn_overbet_monster_raises():
    """Monster should raise vs turn overbet."""
    d = decide("7h7d", "7s2cKd5h", pot=10.0, to_call=12.0, sizing_ratio=1.2)
    assert d.action in ("raise", "all-in"), f"Expected raise with set vs turn overbet, got {d.action}"


def test_turn_overbet_marginal_folds():
    """Marginal hand without strong equity should fold vs turn overbet."""
    d = decide("5h4s", "Ah8d3cKh", pot=10.0, to_call=12.0, sizing_ratio=1.2)
    assert d.action == "fold", f"Expected fold with marginal hand vs turn overbet, got {d.action}"


# ---------------------------------------------------------------------------
# Turn draw equity (1 card to come)
# ---------------------------------------------------------------------------

def test_turn_flush_draw_calls_with_good_odds():
    """Flush draw on turn should call when pot odds are favorable (1 card to come ≈ 19.6%)."""
    # Flush draw equity 1 card to come ≈ 19.6%, pot odds < 30%
    d = decide("AhQh", "9h6h2d7s", pot=10.0, to_call=2.0, sizing_ratio=0.2)
    assert d.action in ("call", "raise"), f"Expected call/raise with flush draw vs small bet, got {d.action}"


def test_turn_oesd_calls_good_odds():
    """OESD on turn should call when pot odds are favorable (1 card to come ≈ 17.4%)."""
    # OESD: 4 5 6 7 → need 3 or 8
    d = decide("5h6d", "4s7c2hKd", pot=10.0, to_call=2.0, sizing_ratio=0.2)
    assert d.action in ("call", "raise"), f"Expected call/raise with OESD vs small bet, got {d.action}"


def test_turn_draw_folds_bad_odds():
    """Draw should fold when pot odds are too expensive (e.g., facing 90% pot bet)."""
    # Gutshot only, facing large bet
    d = decide("5h9d", "As8d3cKh", pot=10.0, to_call=9.0, sizing_ratio=0.9)
    assert d.action == "fold", f"Expected fold with gutshot facing large turn bet, got {d.action}"
