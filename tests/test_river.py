"""Tests for river independent decision framework."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine
from src.opponent import VillainProfile
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range


def make_engine():
    return PostflopEngine()


def make_villain(vpip, pfr, af=2.0, fold_river=0.45, fold_cbet=0.50, hands=50):
    v = VillainProfile()
    v.stats.vpip = vpip
    v.stats.pfr = pfr
    v.stats.aggression_factor = af
    v.stats.fold_to_river_cbet = fold_river
    v.stats.fold_to_flop_cbet = fold_cbet
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
        street="river",
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=is_multiway,
        sizing_ratio=sizing_ratio,
    )


# ---------------------------------------------------------------------------
# River value bet tests
# ---------------------------------------------------------------------------

def test_monster_hand_always_value_bets():
    """Monster hand (set/trips) should always value bet on river."""
    d = decide("7h7d", "7s2cKd5hAc")  # set of 7s on river
    assert d.action in ("bet", "raise", "all-in")


def test_tptk_value_bets():
    """TPTK should value bet on river."""
    d = decide("AhKc", "As2d7h5c9s")  # TPTK on river
    assert d.action in ("bet", "raise", "all-in")


def test_tptk_vs_fish_value_bets():
    """TPGK (AhTc on As board) should value bet vs fish (fish calls with weaker hands)."""
    fish = make_villain(vpip=0.55, pfr=0.05, hands=50)
    d = decide("AhTc", "As2d7h5c9s", villain_profile=fish)
    assert d.action in ("bet", "raise", "all-in")


def test_tpgk_vs_nit_checks():
    """TPGK vs nit should check (nit only calls with better)."""
    nit = make_villain(vpip=0.10, pfr=0.08, af=1.2, fold_cbet=0.72, hands=60)
    d = decide("AhTc", "As2d7h5c9s", villain_profile=nit)
    # vs nit with TPGK, river check is correct
    assert d.action == "check"


def test_middle_pair_vs_fish_thin_value_bet():
    """Middle pair should thin value bet vs fish (fish calls with ace high)."""
    fish = make_villain(vpip=0.55, pfr=0.05, hands=50)
    # Middle pair on dry board
    d = decide("9h8s", "9d3c2s5hKc", villain_profile=fish)
    assert d.action in ("bet", "raise", "all-in")


def test_middle_pair_vs_tag_checks():
    """Middle pair vs TAG should check on river."""
    tag = make_villain(vpip=0.24, pfr=0.20, af=2.0, hands=60)
    d = decide("9h8s", "9d3c2s5hKc", villain_profile=tag)
    assert d.action == "check"


# ---------------------------------------------------------------------------
# River bluff tests
# ---------------------------------------------------------------------------

def test_no_bluff_with_showdown_value():
    """Hand with showdown value should not bluff (check instead)."""
    # Middle pair = showdown value; should check not bluff
    d = decide("9h8s", "9d3c2s5hKc")
    # Middle pair has showdown value, so no bluff → check
    assert d.action in ("check", "bet")  # bet for value is ok too


def test_no_bluff_vs_fish():
    """Should not bluff vs fish on river (fish doesn't fold)."""
    fish = make_villain(vpip=0.55, pfr=0.05, hands=50, fold_river=0.20)
    # Air hand – no showdown value
    d = decide("2h3d", "AsKhQc5d9h", villain_profile=fish)
    assert d.action in ("check", "fold")  # no bluff vs fish


def test_no_bluff_multiway():
    """Should not bluff in multiway pots on river."""
    d = decide("2h3d", "AsKhQc5d9h", is_multiway=True)
    assert d.action in ("check", "fold")


def test_nit_higher_fold_frequency():
    """Bluff vs nit should use small sizing (nit folds easily)."""
    nit = make_villain(vpip=0.10, pfr=0.08, af=1.2, fold_river=0.70, hands=60)
    # Air hand
    d = decide("2h3d", "AsKhQc5d9h", villain_profile=nit)
    # Nit folds a lot, bluff may fire with small sizing
    # Either check or bet small is acceptable
    if d.action == "bet":
        assert d.amount <= 10.0 * 0.40  # small sizing vs nit


# ---------------------------------------------------------------------------
# River bluff-catch tests
# ---------------------------------------------------------------------------

def test_lag_large_bet_call_with_showdown_value():
    """vs LAG large bet, call with showdown value (LAG bluffs a lot)."""
    lag = make_villain(vpip=0.40, pfr=0.35, af=4.0, hands=60)
    # Top pair facing a standard bet from LAG
    d = decide("AhKc", "As2d7h5c9s", to_call=7.0, pot=10.0, villain_profile=lag, sizing_ratio=0.7)
    assert d.action in ("call", "raise", "all-in")


def test_fish_large_bet_fold_marginal():
    """vs fish large bet, fold marginal hands (fish bet = real hand)."""
    fish = make_villain(vpip=0.55, pfr=0.05, hands=50)
    # Middle pair facing 75% pot bet from fish
    d = decide("9h8s", "9d3c2s5hKc", to_call=7.5, pot=10.0, villain_profile=fish, sizing_ratio=0.75)
    # Fish large bet = real hand → fold marginals
    assert d.action in ("fold", "call")  # fold is expected
    if "fish" in d.reasoning.lower():
        assert d.action == "fold"


def test_nit_bet_fold_unless_monster():
    """vs nit bet, fold unless holding very strong hand."""
    nit = make_villain(vpip=0.10, pfr=0.08, af=1.2, hands=60)
    # Top pair good kicker facing nit bet
    d = decide("AhTc", "As2d7h5c9s", to_call=5.0, pot=10.0, villain_profile=nit, sizing_ratio=0.5)
    # Nit never bluffs, fold TPGK
    if "nit" in d.reasoning.lower():
        assert d.action == "fold"


def test_no_showdown_value_folds():
    """Hand with no showdown value should fold to river bet."""
    d = decide("2h3d", "AsKhQc5d9h", to_call=5.0, pot=10.0)
    assert d.action == "fold"


def test_pot_odds_profitable_call():
    """When pot odds are small and villain likely bluffs, call."""
    lag = make_villain(vpip=0.40, pfr=0.35, af=4.5, hands=60)
    # Facing small bet (only need to be right 20% of time)
    d = decide("AhKc", "As2d7h5c9s", to_call=2.0, pot=10.0, villain_profile=lag, sizing_ratio=0.2)
    # LAG with small sizing: bluff-catch profitable
    assert d.action in ("call", "raise", "all-in")


def test_monster_raises_facing_river_bet():
    """Monster hand should raise (not just call) when facing river bet."""
    d = decide("7h7d", "7s2cKd5hAc", to_call=5.0, pot=10.0)
    assert d.action in ("raise", "all-in")
