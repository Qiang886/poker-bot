"""GTO-inspired preflop range charts for 6-max 100BB."""

from typing import Set
from src.position import Position


# ---------------------------------------------------------------------------
# RFI (Raise First In) ranges
# ---------------------------------------------------------------------------

_UTG_RFI: Set[str] = {
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s",
    "QJs", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "98s", "87s", "76s", "65s",
    "AKo", "AQo", "AJo",
}

_HJ_RFI: Set[str] = _UTG_RFI | {
    "66", "55",
    "A8s", "A7s", "A6s",
    "K8s", "K7s",
    "Q8s",
    "J8s",
    "T8s", "97s", "86s", "75s", "64s", "54s",
    "ATo", "KQo",
}

_CO_RFI: Set[str] = _HJ_RFI | {
    "44", "33", "22",
    "K6s", "K5s", "K4s",
    "Q7s", "Q6s",
    "J7s",
    "T7s", "96s", "85s", "74s", "63s", "53s", "43s",
    "AJo", "ATo", "A9o", "KJo", "QJo",
}

_BTN_RFI: Set[str] = _CO_RFI | {
    "K3s", "K2s",
    "Q5s", "Q4s", "Q3s", "Q2s",
    "J6s", "J5s", "J4s", "J3s", "J2s",
    "T6s", "T5s", "T4s",
    "95s", "84s", "73s", "62s", "52s", "42s", "32s",
    "A8o", "A7o", "A6o", "A5o",
    "KTo", "K9o", "QTo", "Q9o", "JTo", "J9o", "T9o",
}

_SB_RFI: Set[str] = _BTN_RFI | {
    "T3s", "T2s",
    "94s", "93s", "92s",
    "83s", "82s", "72s",
    "A4o", "A3o", "A2o",
    "K8o", "K7o", "Q8o",
}


def get_rfi_range(position: Position) -> Set[str]:
    mapping = {
        Position.UTG: _UTG_RFI,
        Position.HJ: _HJ_RFI,
        Position.CO: _CO_RFI,
        Position.BTN: _BTN_RFI,
        Position.SB: _SB_RFI,
        Position.BB: set(),  # BB never RFIs preflop (would be a limp)
    }
    return mapping[position]


# ---------------------------------------------------------------------------
# 3-bet ranges (value + polar bluffs), keyed by (hero_pos, villain_rfi_pos)
# ---------------------------------------------------------------------------

_3BET_VALUE_CORE: Set[str] = {"AA", "KK", "QQ", "AKs", "AKo"}

_3BET_BLUFFS_COMMON: Set[str] = {
    "A5s", "A4s", "A3s", "A2s",
    "K5s", "K4s",
    "76s", "65s", "54s",
}

def _build_3bet(extra_value: Set[str] = frozenset(),
                extra_bluffs: Set[str] = frozenset()) -> Set[str]:
    return _3BET_VALUE_CORE | extra_value | _3BET_BLUFFS_COMMON | extra_bluffs


_3BET_RANGES = {
    # (hero, villain_rfi)
    (Position.BB, Position.UTG): _build_3bet({"JJ", "TT"}),
    (Position.BB, Position.HJ): _build_3bet({"JJ", "TT", "AQs"}),
    (Position.BB, Position.CO): _build_3bet({"JJ", "TT", "99", "AQs", "AJs"}),
    (Position.BB, Position.BTN): _build_3bet({"JJ", "TT", "99", "88", "AQs", "AJs", "ATs", "KQs"},
                                              {"Q5s", "J5s", "T5s"}),
    (Position.BB, Position.SB): _build_3bet({"JJ", "TT", "99", "88", "AQs", "AJs", "ATs", "KQs"},
                                              {"Q4s", "J4s", "T4s"}),
    (Position.SB, Position.UTG): _build_3bet({"JJ", "TT"}),
    (Position.SB, Position.HJ): _build_3bet({"JJ", "TT", "AQs"}),
    (Position.SB, Position.CO): _build_3bet({"JJ", "TT", "99", "AQs", "AJs"}),
    (Position.SB, Position.BTN): _build_3bet({"JJ", "TT", "99", "88", "AQs", "AJs", "ATs"}),
    (Position.BTN, Position.UTG): _build_3bet({"JJ"}),
    (Position.BTN, Position.HJ): _build_3bet({"JJ", "TT", "AQs"}),
    (Position.BTN, Position.CO): _build_3bet({"JJ", "TT", "99", "AQs", "AJs"}),
    (Position.BTN, Position.SB): _build_3bet({"JJ", "TT", "99", "AQs", "AJs"}),
    (Position.CO, Position.UTG): _build_3bet(),
    (Position.CO, Position.HJ): _build_3bet({"JJ", "TT"}),
    (Position.CO, Position.SB): _build_3bet({"JJ", "TT", "99", "AQs"}),
    (Position.HJ, Position.UTG): _build_3bet(),
    (Position.HJ, Position.SB): _build_3bet({"JJ", "TT", "AQs"}),
    (Position.UTG, Position.SB): _build_3bet(),
}

_DEFAULT_3BET = _build_3bet()


def get_3bet_range(hero_pos: Position, villain_pos: Position) -> Set[str]:
    return _3BET_RANGES.get((hero_pos, villain_pos), _DEFAULT_3BET)


# ---------------------------------------------------------------------------
# 4-bet ranges
# ---------------------------------------------------------------------------

_4BET_RANGES = {
    Position.UTG: {"AA", "KK", "QQ", "AKs", "AKo", "A5s", "A4s"},
    Position.HJ:  {"AA", "KK", "QQ", "AKs", "AKo", "A5s", "A4s", "JJ"},
    Position.CO:  {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "A5s", "A4s", "A3s"},
    Position.BTN: {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "A5s", "A4s", "A3s", "A2s"},
    Position.SB:  {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "A5s", "A4s"},
    Position.BB:  {"AA", "KK", "QQ", "AKs", "AKo", "A5s", "A4s"},
}


def get_4bet_range(hero_pos: Position) -> Set[str]:
    return _4BET_RANGES.get(hero_pos, {"AA", "KK", "AKs", "AKo"})


# ---------------------------------------------------------------------------
# Call-vs-RFI ranges
# ---------------------------------------------------------------------------

_CALL_RFI = {
    # (hero_pos, villain_rfi_pos)
    (Position.BB, Position.UTG): {
        "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s",
        "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
        "AQo", "AJo", "ATo", "KQo", "KJo",
    },
    (Position.BB, Position.HJ): {
        "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "QJs", "QTs", "Q9s", "Q8s", "JTs", "J9s", "T9s", "T8s", "98s", "97s",
        "87s", "86s", "76s", "75s", "65s", "64s", "54s",
        "AQo", "AJo", "ATo", "A9o", "KQo", "KJo", "QJo",
    },
    (Position.BB, Position.CO): {
        "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
        "QJs", "QTs", "Q9s", "Q8s", "JTs", "J9s", "J8s", "T9s", "T8s",
        "98s", "97s", "87s", "86s", "76s", "75s", "65s", "64s", "54s", "53s",
        "AQo", "AJo", "ATo", "A9o", "A8o",
        "KQo", "KJo", "KTo", "QJo", "QTo",
    },
    (Position.BB, Position.BTN): {
        "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "JTs", "J9s", "J8s", "J7s",
        "T9s", "T8s", "T7s",
        "98s", "97s", "96s", "87s", "86s", "85s",
        "76s", "75s", "74s", "65s", "64s", "54s", "53s", "43s",
        "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o",
        "KQo", "KJo", "KTo", "K9o",
        "QJo", "QTo", "Q9o", "JTo", "J9o", "T9o",
    },
    (Position.BB, Position.SB): {
        "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s",
        "98s", "97s", "87s", "86s", "76s", "75s", "65s", "64s", "54s", "53s",
        "AQo", "AJo", "ATo", "A9o", "A8o", "A7o",
        "KQo", "KJo", "KTo", "K9o",
        "QJo", "QTo", "Q9o", "JTo", "J9o", "T9o",
    },
    (Position.SB, Position.BTN): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
        "AJo", "ATo", "KQo", "KJo",
    },
    (Position.BTN, Position.CO): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AJs", "ATs", "A9s", "A8s", "A7s", "A6s",
        "KQs", "KJs", "KTs", "K9s",
        "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
        "AJo", "ATo", "A9o", "KQo", "KJo",
    },
    (Position.BTN, Position.HJ): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs", "A9s", "A8s",
        "KQs", "KJs", "KTs",
        "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
        "AJo", "ATo", "KQo",
    },
    (Position.BTN, Position.UTG): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs",
        "KQs", "KJs", "KTs",
        "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
        "AJo", "KQo",
    },
    (Position.CO, Position.HJ): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AJs", "ATs", "A9s", "A8s",
        "KQs", "KJs", "KTs",
        "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
        "AJo", "ATo", "KQo",
    },
    (Position.CO, Position.UTG): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs",
        "KQs", "KJs",
        "QJs", "JTs", "T9s", "98s", "87s",
        "AJo", "KQo",
    },
    (Position.HJ, Position.UTG): {
        "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AQs", "AJs", "ATs",
        "KQs", "KJs",
        "QJs", "JTs", "T9s", "98s",
        "AJo",
    },
}

_DEFAULT_CALL_RFI: Set[str] = {
    "TT", "99", "88", "77", "66", "55",
    "AQs", "AJs", "ATs",
    "KQs", "KJs",
    "QJs", "JTs",
}


def get_call_rfi_range(hero_pos: Position, villain_pos: Position) -> Set[str]:
    return _CALL_RFI.get((hero_pos, villain_pos), _DEFAULT_CALL_RFI)


# ---------------------------------------------------------------------------
# Call-vs-3bet ranges
# ---------------------------------------------------------------------------

_CALL_3BET: Set[str] = {
    "QQ", "JJ", "TT", "99", "88",
    "AQs", "AJs", "ATs",
    "KQs", "KJs",
    "QJs", "JTs", "T9s",
    "AQo", "KQo",
}

_CALL_3BET_IP: Set[str] = _CALL_3BET | {
    "77", "66", "55",
    "A9s", "A8s",
    "KTs",
    "QTs", "98s", "87s",
}


def get_call_3bet_range(hero_pos: Position, villain_pos: Position) -> Set[str]:
    in_position = (
        (hero_pos == Position.BTN and villain_pos in (Position.CO, Position.HJ, Position.UTG, Position.SB)) or
        (hero_pos == Position.CO and villain_pos in (Position.HJ, Position.UTG)) or
        (hero_pos == Position.HJ and villain_pos == Position.UTG)
    )
    return _CALL_3BET_IP if in_position else _CALL_3BET


# ---------------------------------------------------------------------------
# Squeeze ranges
# ---------------------------------------------------------------------------

_SQUEEZE_RANGES = {
    Position.BB:  {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "A5s", "A4s", "A3s"},
    Position.SB:  {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "A5s", "A4s"},
    Position.BTN: {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "A5s", "A4s"},
    Position.CO:  {"AA", "KK", "QQ", "AKs", "AKo", "AQs", "A5s"},
    Position.HJ:  {"AA", "KK", "QQ", "AKs", "AKo"},
    Position.UTG: {"AA", "KK", "AKs"},
}


def get_squeeze_range(hero_pos: Position) -> Set[str]:
    return _SQUEEZE_RANGES.get(hero_pos, {"AA", "KK", "AKs", "AKo"})
