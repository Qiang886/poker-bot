"""Tests for the hand evaluator."""

import pytest
from src.card import card_from_str, cards_from_str
from src.evaluator import evaluate_5, evaluate_7, HandRank


def make(s): return cards_from_str(s)


def test_royal_flush_beats_straight_flush():
    royal = make("AsTsKsQsJs")
    sf = make("9s8s7s6s5s")
    assert evaluate_5(royal) > evaluate_5(sf)


def test_straight_flush_beats_four_of_a_kind():
    sf = make("9s8s7s6s5s")
    quads = make("AsAhAdAc2s")
    assert evaluate_5(sf) > evaluate_5(quads)


def test_four_of_a_kind_beats_full_house():
    quads = make("AsAhAdAc2s")
    fh = make("AsAhAdKsKh")
    assert evaluate_5(quads) > evaluate_5(fh)


def test_full_house_beats_flush():
    fh = make("AsAhAdKsKh")
    flush = make("2s5s9sJs3s")
    assert evaluate_5(fh) > evaluate_5(flush)


def test_flush_beats_straight():
    flush = make("2s5s9sJs3s")
    straight = make("9hTsJdQcKh")
    assert evaluate_5(flush) > evaluate_5(straight)


def test_straight_beats_three_of_a_kind():
    straight = make("9hTsJdQcKh")
    trips = make("AsAhAdKs2c")
    assert evaluate_5(straight) > evaluate_5(trips)


def test_three_beats_two_pair():
    trips = make("AsAhAdKs2c")
    two_pair = make("AsAhKdKs2c")
    assert evaluate_5(trips) > evaluate_5(two_pair)


def test_two_pair_beats_one_pair():
    two_pair = make("AsAhKdKs2c")
    one_pair = make("AsAh2d3s4c")
    assert evaluate_5(two_pair) > evaluate_5(one_pair)


def test_one_pair_beats_high_card():
    one_pair = make("AsAh2d3s4c")
    high_card = make("AsKh2d3s5c")
    assert evaluate_5(one_pair) > evaluate_5(high_card)


def test_evaluate_7_returns_best_hand():
    # 7 cards containing a flush among them
    cards = cards_from_str("AsKsQsJsTs2h3d")
    rank, tb = evaluate_7(cards)
    assert rank == HandRank.STRAIGHT_FLUSH


def test_ace_low_straight():
    cards = make("Ac2d3h4s5c")
    rank, tb = evaluate_5(cards)
    assert rank == HandRank.STRAIGHT
    assert tb == (5,)


def test_hand_rank_enum_ordering():
    assert HandRank.STRAIGHT_FLUSH > HandRank.FOUR_OF_A_KIND
    assert HandRank.FOUR_OF_A_KIND > HandRank.FULL_HOUSE
    assert HandRank.FULL_HOUSE > HandRank.FLUSH
    assert HandRank.FLUSH > HandRank.STRAIGHT
    assert HandRank.STRAIGHT > HandRank.THREE_OF_A_KIND
    assert HandRank.THREE_OF_A_KIND > HandRank.TWO_PAIR
    assert HandRank.TWO_PAIR > HandRank.ONE_PAIR
    assert HandRank.ONE_PAIR > HandRank.HIGH_CARD
