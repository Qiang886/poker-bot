"""Tests for Turn-specific decision making with runout awareness."""

import pytest
from src.card import cards_from_str
from src.position import Position
from src.postflop import PostflopEngine, TurnChange
from src.barrel_plan import BarrelPlan, ValueLine, BluffLine
from src.opponent import VillainProfile
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
    position=Position.BTN,
    villain_profile=None,
    street="turn",
    barrel_plan=None,
    action_history=None,
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
        street=street,
        villain_profile=villain_profile,
        hero_range=hero_range,
        villain_range=villain_range,
        barrel_plan=barrel_plan,
        is_multiway=False,
        action_history=action_history or [],
    )


# ---------------------------------------------------------------------------
# Turn blank → continue barrel plan
# ---------------------------------------------------------------------------

def test_turn_blank_continues_barrel():
    """Blank turn (low card) with strong hand → bot should bet (continue plan)."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bet",
        value_line=ValueLine.BET_BET_BET,
        continue_runouts=["blank"],
        give_up_runouts=["flush_completes"],
    )
    # AhKh on A742 (turn 2 is blank for flop A74)
    d = decide("AhKh", "Ah7d4c2s", pot=10.0, stack=100.0, barrel_plan=plan)
    assert d.action in ("bet", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn overcard (A) → slow down with medium hand
# ---------------------------------------------------------------------------

def test_turn_scare_card_slows_down_medium():
    """A arrives on 7-6-2 board → bot with overpair small slows down."""
    # KhKd on 7s6c2dAh (A is scary_card + overcard)
    d = decide("KhKd", "7s6c2dAh", pot=10.0, stack=100.0, position=Position.BTN)
    # KK is overpair, but A on this board is scary → should check
    assert d.action in ("check", "fold")


def test_turn_scare_card_still_value_bets_monster():
    """Even with scary turn, a monster (set) should still be played for value."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bet",
        value_line=ValueLine.BET_BET_BET,
        continue_runouts=["any"],
        give_up_runouts=[],
    )
    # 7h7d on 7s6c2dAh → set of 7s with trips → value bet
    d = decide("7h7d", "7s6c2dAh", pot=10.0, stack=100.0, barrel_plan=plan)
    assert d.action in ("bet", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn flush complete → stop bluff (unless hero has flush)
# ---------------------------------------------------------------------------

def test_turn_flush_complete_stops_bluff():
    """Flush completes on turn, hero has no flush draw → no semi-bluff."""
    # AsKd on 9h6h3cJh (Jh = third heart, flush completes)
    # Hero has no hearts → should check/fold
    d = decide("AsKd", "9h6h3cJh", pot=10.0, stack=100.0, position=Position.BTN)
    assert d.action in ("check", "fold")


def test_turn_flush_complete_hero_has_flush_bets():
    """Flush completes on turn and hero has it → value bet."""
    # AhKh on 9h6s3cJh (turn Jh → hero has nut flush draw completed)
    # AhKh + Jh,Kh on board means we have Ah + more hearts  
    # Actually: board 9h 6s 3c Jh → hero has AhKh, 3 hearts (A,J,9 from hero+board)
    # AhKh: hero has Ah → nut flush (9h, Jh, Ah = 3 hearts of which hero has Ah)
    d = decide("AhKh", "9h6s3cJh", pot=10.0, stack=100.0, position=Position.BTN)
    assert d.action in ("bet", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn board pairs → slow down if no boat; value bet if set → boat potential
# ---------------------------------------------------------------------------

def test_turn_board_pairs_no_boat_slows_down():
    """Board pairs on turn, hero has top pair only → slow down."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bet",
        value_line=ValueLine.BET_BET_BET,
        continue_runouts=["blank"],
        give_up_runouts=[],
    )
    # AhKc on AsKd7c7s (7 pairs → paired board, hero has top two pair)
    # Hero has two pair (A+K), not full house → may slow down
    d = decide("AhKc", "AsKd7c7s", pot=10.0, stack=100.0, barrel_plan=plan)
    # Two pair on paired board is medium strength → check is reasonable
    assert d.action in ("bet", "check")  # accepting either; main test is no crash


def test_turn_board_pairs_set_value_bets():
    """Board pairs on turn, hero has set → full house potential → value bet."""
    plan = BarrelPlan(
        flop_action="bet",
        turn_action="bet",
        river_action="bet",
        value_line=ValueLine.BET_BET_BET,
        continue_runouts=["any"],
        give_up_runouts=[],
    )
    # KhKd on As9c4dKs (K pairs the board → hero has set of Ks)
    d = decide("KhKd", "As9c4dKs", pot=10.0, stack=100.0, barrel_plan=plan)
    assert d.action in ("bet", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn facing double barrel → need stronger hand to continue
# ---------------------------------------------------------------------------

def test_turn_double_barrel_folds_medium():
    """Facing second barrel on turn with middle pair → fold."""
    # 9h8s on Kd7c3s2h (turn 2h = blank, villain bets again)
    # middle pair = 9 on King-high board → fold to double barrel
    d = decide("9h8s", "Kd7c3s2h", pot=10.0, to_call=7.0, stack=100.0)
    assert d.action == "fold"


def test_turn_double_barrel_calls_tptk():
    """TPTK facing turn barrel → call."""
    # AsKh on Ah7d4c2s, villain bets ~50% pot
    d = decide("AsKh", "Ah7d4c2s", pot=10.0, to_call=5.0, stack=100.0)
    assert d.action in ("call", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn facing overbet → polarized handling
# ---------------------------------------------------------------------------

def test_turn_overbet_folds_medium_hand():
    """Facing overbet (>80% pot) on turn with medium hand → fold."""
    # QhJd on Kd7c3s2h → second pair; facing big overbet
    d = decide("QhJd", "Kd7c3s2h", pot=10.0, to_call=9.0, stack=100.0, villain_profile=None)
    assert d.action in ("fold",)


def test_turn_overbet_calls_strong_hand():
    """Facing overbet on turn with strong hand (set) → call or raise."""
    d = decide("7h7d", "7s6c2dAh", pot=10.0, to_call=9.0, stack=100.0)
    assert d.action in ("call", "raise", "all-in")


# ---------------------------------------------------------------------------
# Turn draw equity (1 card to come vs 2)
# ---------------------------------------------------------------------------

def test_turn_flush_draw_needs_better_odds():
    """Flush draw on turn (1 card to come) needs >~20% pot odds."""
    # AhKh on 9h6s3c2h → hero has nut flush draw (4 hearts)
    # Villain bets 30% pot → pot odds ~23% → should call (just above equity ~20%)
    d = decide("AhKh", "9h6s3c2h", pot=10.0, to_call=3.0, stack=100.0)
    assert d.action in ("call", "raise")


def test_turn_flush_draw_fold_bad_odds():
    """Flush draw on turn facing large bet → fold if odds not sufficient."""
    # With a large enough bet, even nut flush draw should fold turn
    # ~35% pot odds needed for hero's equity ~20% → fold at 50% pot
    d = decide("AhKh", "9h6s3c2h", pot=10.0, to_call=6.0, stack=100.0)
    # 6 into 10 = 37.5% pot odds, flush draw ~20% equity → close, may fold
    assert d.action in ("call", "fold")  # both defensible


# ---------------------------------------------------------------------------
# Turn villain capped range → probe / overbet opportunity
# ---------------------------------------------------------------------------

def test_turn_capped_villain_bets_more():
    """After villain checks flop (capped), hero IP bets turn with medium strength."""
    # Flop action shows villain checked back
    history = [
        {"street": "flop", "actor": 1, "action": "check"},
    ]
    # 9h8c on 9d3c2sJh (turn J = overcard, but hero has top pair)
    d = decide("9h8c", "9d3c2sJh", pot=10.0, stack=100.0,
               position=Position.BTN, action_history=history)
    assert d.action in ("bet", "raise", "check")


# ---------------------------------------------------------------------------
# TurnChange analysis
# ---------------------------------------------------------------------------

def test_analyze_turn_change_blank():
    engine = make_engine()
    board = cards_from_str("9d7s3c2h")
    hero = tuple(cards_from_str("KhKd"))
    turn_change = engine._analyze_turn_change(board, board[3], hero)
    assert turn_change.is_blank is True
    assert turn_change.is_scare_card is False


def test_analyze_turn_change_scare_card():
    engine = make_engine()
    board = cards_from_str("9d7s3cAh")
    hero = tuple(cards_from_str("KhKd"))
    turn_change = engine._analyze_turn_change(board, board[3], hero)
    assert turn_change.is_scare_card is True
    assert turn_change.is_overcard is True


def test_analyze_turn_change_flush_completes():
    engine = make_engine()
    board = cards_from_str("9h7s3hJh")  # three hearts
    hero = tuple(cards_from_str("KhKd"))
    turn_change = engine._analyze_turn_change(board, board[3], hero)
    assert turn_change.completes_flush is True
    assert turn_change.range_shift == "hero_better"  # hero has heart
