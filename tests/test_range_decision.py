"""Tests for villain_range participation in postflop decisions."""

import pytest
from src.card import cards_from_str, Card
from src.position import Position
from src.postflop import PostflopEngine
from src.hand_analysis import MadeHandType, classify_hand
from src.weighted_range import ComboWeight, build_range_combos
from src.ranges import get_rfi_range
from src.opponent import VillainProfile


def make_engine():
    return PostflopEngine()


def make_villain(vpip=0.28, pfr=0.22, af=2.0, hands=60):
    v = VillainProfile()
    v.stats.vpip = vpip
    v.stats.pfr = pfr
    v.stats.aggression_factor = af
    v.stats.hands_played = hands
    return v


def make_combo_weight(combo_str, weight=1.0):
    cards = cards_from_str(combo_str)
    return ComboWeight(combo=tuple(cards), weight=weight)


def decide_river(
    hand_str,
    board_str,
    pot=10.0,
    to_call=0.0,
    stack=100.0,
    villain_profile=None,
    villain_range=None,
    sizing_ratio=0.0,
):
    engine = make_engine()
    hand = tuple(cards_from_str(hand_str))
    board = cards_from_str(board_str)
    dead = list(hand) + board
    if villain_range is None:
        villain_range = []
    hero_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    return engine.decide(
        hero_hand=hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=stack,
        villain_stack=stack,
        position=Position.BTN,
        street="river",
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=None,
        is_multiway=False,
        sizing_ratio=sizing_ratio,
    )


# ---------------------------------------------------------------------------
# thin value bet tests
# ---------------------------------------------------------------------------

def test_villain_range_weak_enables_thin_value():
    """When villain_range has many weak hands (BOTTOM_PAIR), thin value bet is enabled."""
    board = cards_from_str("9d3c2s5hKc")
    # Create a villain range full of bottom-pair or weaker hands
    # Use combos that make pair of 2s or pair of 3s (weaker than pair of 9s)
    weak_combos = [
        make_combo_weight("2c2h", 1.0),   # trips on board
        make_combo_weight("3h3s", 1.0),   # trips on board
        make_combo_weight("4h4c", 1.0),   # pair of 4s < pair of 9s
        make_combo_weight("5h5c", 1.0),   # pair of 5s
        make_combo_weight("6h6c", 1.0),   # pair of 6s
        make_combo_weight("7h7c", 1.0),   # pair of 7s
        make_combo_weight("8h8c", 1.0),   # pair of 8s
    ]
    # hero has pair of 9s (MIDDLE_PAIR) — TAG vs weak range should still check
    # Note: 2c2h and 3h3s actually make trips which are STRONGER than pair of 9s
    # Use weaker made hands: pairs of 4-8
    weak_combos_real = [
        make_combo_weight("4h4c", 1.0),   # pair of 4s
        make_combo_weight("5h5c", 1.0),   # pair of 5s
        make_combo_weight("6h6c", 1.0),   # pair of 6s
        make_combo_weight("7h7c", 1.0),   # pair of 7s
        make_combo_weight("8h8c", 1.0),   # pair of 8s (all weaker than 9s)
    ]
    tag = make_villain(vpip=0.28, pfr=0.22, af=2.0, hands=60)
    engine = make_engine()
    can_value, sizing = engine._can_thin_value_bet(
        classify_hand(tuple(cards_from_str("9h8s")), board),
        tag,
        "TAG",
        board,
        pot=10.0,
        hero_stack=100.0,
        spr=10.0,
        villain_range=weak_combos_real,
    )
    # All 5 combos are weaker than pair of 9s (with showdown value) → > 30% threshold → can value
    assert can_value is True


def test_villain_range_strong_no_thin_value():
    """When villain_range has mainly strong hands (TOP_PAIR+), thin value bet is disabled."""
    board = cards_from_str("9d3c2s5hKc")
    # Combos that pair the K (TOP_PAIR) or better → all stronger than pair of 9s
    strong_combos = [
        make_combo_weight("KhJc", 1.0),   # pairs K = TOP_PAIR_GOOD_KICKER
        make_combo_weight("KcQh", 1.0),   # pairs K = TOP_PAIR_GOOD_KICKER
        make_combo_weight("KdJd", 1.0),   # pairs K
        make_combo_weight("AsAh", 1.0),   # overpair AA (no As on board)
        make_combo_weight("AhAc", 1.0),   # overpair AA
    ]
    tag = make_villain(vpip=0.28, pfr=0.22, af=2.0, hands=60)
    engine = make_engine()
    can_value, sizing = engine._can_thin_value_bet(
        classify_hand(tuple(cards_from_str("9h8s")), board),
        tag,
        "TAG",
        board,
        pot=10.0,
        hero_stack=100.0,
        spr=10.0,
        villain_range=strong_combos,
    )
    # All combos pair K or are AA overpair → all stronger than pair of 9s → can't thin value
    assert can_value is False


def test_villain_range_empty_uses_type_fallback():
    """Empty villain_range should fall back to player-type logic."""
    board = cards_from_str("9d3c2s5hKc")
    fish = make_villain(vpip=0.55, pfr=0.05, hands=60)
    engine = make_engine()
    # Fish: MIDDLE_PAIR should value bet (fish calls with worse)
    can_value, sizing = engine._can_thin_value_bet(
        classify_hand(tuple(cards_from_str("9h8s")), board),
        fish,
        "fish",
        board,
        pot=10.0,
        hero_stack=100.0,
        spr=10.0,
        villain_range=[],  # empty range
    )
    assert can_value is True
    assert sizing > 0


def test_bluff_catch_high_range_bluff_calls():
    """When villain range has many weak combos (likely bluffs), bluff-catch should call."""
    board = cards_from_str("AsKd7h2c9s")
    # Range is mostly weak (air hands)
    weak_bluff_range = [
        make_combo_weight("2h3h", 1.0),   # no pair
        make_combo_weight("4c5c", 1.0),   # no pair
        make_combo_weight("6h8d", 1.0),   # no pair
        make_combo_weight("JhTh", 1.0),   # no pair
        make_combo_weight("QhJh", 1.0),   # no pair
    ]
    engine = make_engine()
    should_call, reasoning = engine._river_bluff_catch(
        classify_hand(tuple(cards_from_str("AhTc")), board),
        None,
        "TAG",
        weak_bluff_range,
        board,
        pot=10.0,
        to_call=4.0,
        sizing_ratio=0.4,
    )
    # High fraction of air → high bluff estimate → should call
    assert should_call is True
    assert "range_bluff" in reasoning


def test_bluff_catch_empty_range_uses_type_estimate():
    """Empty villain_range should fall back to player-type bluff estimate."""
    board = cards_from_str("AsKd7h2c9s")
    engine = make_engine()
    should_call_lag, reasoning_lag = engine._river_bluff_catch(
        classify_hand(tuple(cards_from_str("AhTc")), board),
        None,
        "LAG",
        [],  # empty range
        board,
        pot=10.0,
        to_call=4.0,
        sizing_ratio=0.7,
    )
    should_call_nit, reasoning_nit = engine._river_bluff_catch(
        classify_hand(tuple(cards_from_str("AhTc")), board),
        None,
        "nit",
        [],  # empty range
        board,
        pot=10.0,
        to_call=4.0,
        sizing_ratio=0.7,
    )
    # LAG bluffs more → more likely to call; nit rarely bluffs → more likely to fold
    # At min, LAG estimated bluff pct > nit estimated bluff pct
    import re
    def extract_pct(s):
        m = re.search(r"estimated_bluff=(\d+)%", s)
        return int(m.group(1)) if m else 0
    lag_pct = extract_pct(reasoning_lag)
    nit_pct = extract_pct(reasoning_nit)
    assert lag_pct > nit_pct


def test_maniac_bluff_catch_higher_estimate():
    """Maniac player type should produce highest bluff estimate."""
    board = cards_from_str("AsKd7h2c9s")
    engine = make_engine()
    _, reasoning = engine._river_bluff_catch(
        classify_hand(tuple(cards_from_str("AhTc")), board),
        None,
        "maniac",
        [],
        board,
        pot=10.0,
        to_call=4.0,
        sizing_ratio=0.7,
    )
    # maniac multiplier is 2.0 → highest
    import re
    m = re.search(r"estimated_bluff=(\d+)%", reasoning)
    assert m and int(m.group(1)) >= 30
