"""Tests for improved opponent classification in VillainProfile."""

import pytest
from src.opponent import VillainProfile


def make_villain(vpip, pfr, af=2.0, wtsd=0.28, hands=50):
    v = VillainProfile()
    v.stats.vpip = vpip
    v.stats.pfr = pfr
    v.stats.aggression_factor = af
    v.stats.wtsd = wtsd
    v.stats.hands_played = hands
    return v


# ---------------------------------------------------------------------------
# is_fish tests
# ---------------------------------------------------------------------------

def test_fish_passive_original_condition():
    """VPIP > 40%, PFR < 15% → fish (original condition)."""
    v = make_villain(vpip=0.45, pfr=0.10)
    assert v.is_fish() is True
    assert v.classify() == "fish"


def test_fish_high_vpip_only():
    """VPIP > 50% alone → fish, regardless of PFR."""
    v = make_villain(vpip=0.55, pfr=0.22)  # Previously would not be detected as fish
    assert v.is_fish() is True
    assert v.classify() == "fish"


def test_fish_calling_station():
    """VPIP > 40% AND WTSD > 35% → calling station (fish).

    Note: make_villain sets wtsd explicitly to 0.40 here to trigger the
    calling-station condition (default wtsd is 0.28, which would not trigger).
    """
    v = make_villain(vpip=0.45, pfr=0.18, wtsd=0.40)
    assert v.is_fish() is True


def test_fish_wider_passive():
    """VPIP > 35% AND PFR < 10% → passive fish (wider condition)."""
    v = make_villain(vpip=0.38, pfr=0.08)
    assert v.is_fish() is True


def test_not_fish_tag():
    """TAG player (VPIP=25%, PFR=20%) should not be fish."""
    v = make_villain(vpip=0.25, pfr=0.20)
    assert v.is_fish() is False


def test_not_fish_lag():
    """LAG player should not be fish (even with VPIP > 40% if PFR is high)."""
    v = make_villain(vpip=0.45, pfr=0.35, af=3.0)
    # VPIP 45% > 40% AND PFR 35% > 15%: NOT original fish condition
    # But VPIP > 35% AND PFR < 10%? No (PFR=35%). VPIP > 50%? No (0.45). → not fish
    # But VPIP > 40% AND WTSD > 0.35? Depends on wtsd default (0.28)
    assert v.is_fish() is False


# ---------------------------------------------------------------------------
# is_maniac tests
# ---------------------------------------------------------------------------

def test_maniac_classification():
    """VPIP > 45%, PFR > 30%, AF > 3.5 → maniac."""
    v = make_villain(vpip=0.50, pfr=0.35, af=4.0)
    assert v.is_maniac() is True
    assert v.classify() == "maniac"


def test_maniac_not_fish_priority():
    """Maniac should be classified before fish in priority (but fish first)."""
    # A player with VPIP=50%, PFR=35%, AF=4.0 could be fish (VPIP>50%)
    # AND maniac. fish takes priority in classify().
    v = make_villain(vpip=0.51, pfr=0.35, af=4.0)
    # VPIP > 50% → fish() is True, classify() returns "fish"
    assert v.classify() == "fish"

    # VPIP=48% (not fish), PFR=35%, AF=4.0 → maniac
    v2 = make_villain(vpip=0.48, pfr=0.35, af=4.0)
    assert v2.is_fish() is False
    assert v2.is_maniac() is True
    assert v2.classify() == "maniac"


def test_not_maniac_low_af():
    """AF <= 3.5 should not be maniac."""
    v = make_villain(vpip=0.48, pfr=0.35, af=3.0)
    assert v.is_maniac() is False


def test_not_maniac_low_pfr():
    """PFR <= 30% should not be maniac."""
    v = make_villain(vpip=0.50, pfr=0.25, af=4.0)
    assert v.is_maniac() is False


# ---------------------------------------------------------------------------
# is_nit tests
# ---------------------------------------------------------------------------

def test_nit_tight_classic():
    """VPIP < 15% → classic nit."""
    v = make_villain(vpip=0.12, pfr=0.10)
    assert v.is_nit() is True
    assert v.classify() == "nit"


def test_nit_improved_threshold():
    """VPIP < 16% → nit (slightly relaxed from 15%)."""
    v = make_villain(vpip=0.155, pfr=0.12)
    assert v.is_nit() is True


def test_nit_limp_passive():
    """VPIP < 20% AND PFR < 12% → passive nit (new condition)."""
    v = make_villain(vpip=0.18, pfr=0.10)  # Occasional limper, never aggressive
    assert v.is_nit() is True
    assert v.classify() == "nit"


def test_not_nit_normal():
    """Normal player (VPIP=25%) is not a nit."""
    v = make_villain(vpip=0.25, pfr=0.18)
    assert v.is_nit() is False


def test_not_nit_active_tag():
    """TAG with VPIP=20%, PFR=16% should not be nit (PFR too high for nit condition)."""
    v = make_villain(vpip=0.20, pfr=0.16)
    # VPIP < 16? No (0.20). VPIP < 20 AND PFR < 12? No (PFR=0.16). → not nit
    assert v.is_nit() is False


# ---------------------------------------------------------------------------
# is_lag tests
# ---------------------------------------------------------------------------

def test_lag_improved_condition():
    """VPIP > 28%, PFR > 22%, AF > 2.5 → LAG (improved, more inclusive)."""
    v = make_villain(vpip=0.30, pfr=0.24, af=2.6)
    assert v.is_lag() is True
    assert v.classify() == "LAG"


def test_not_lag_old_strict_condition():
    """Old strict condition (PFR > 25%, AF > 3.0) excluded some LAGs — new condition includes them."""
    # Old would require VPIP > 30%, PFR > 25%, AF > 3.0
    # New accepts VPIP > 28%, PFR > 22%, AF > 2.5
    v = make_villain(vpip=0.29, pfr=0.23, af=2.6)
    # Old: VPIP > 30%? No → not LAG. New: VPIP > 28% Yes, PFR > 22% Yes, AF > 2.5 Yes → LAG
    assert v.is_lag() is True


def test_not_lag_low_af():
    """AF <= 2.5 → not LAG."""
    v = make_villain(vpip=0.32, pfr=0.25, af=2.0)
    assert v.is_lag() is False


# ---------------------------------------------------------------------------
# classify() priority tests
# ---------------------------------------------------------------------------

def test_classify_fish_before_maniac():
    """Fish detection takes priority over maniac."""
    v = make_villain(vpip=0.55, pfr=0.35, af=4.5)  # Both fish and maniac conditions met
    assert v.classify() == "fish"  # fish wins


def test_classify_maniac_before_nit():
    """Maniac detection should come before nit (maniac > nit in priority)."""
    # Impossible in practice (maniac has high VPIP, nit has low VPIP)
    # but test the logic order
    v = make_villain(vpip=0.48, pfr=0.32, af=4.0)
    assert v.is_nit() is False
    assert v.classify() == "maniac"


def test_classify_tag_default():
    """Player without specific classification → TAG."""
    v = make_villain(vpip=0.25, pfr=0.19, af=2.0)
    assert v.classify() == "TAG"


def test_classify_lag():
    """Standard LAG profile."""
    v = make_villain(vpip=0.35, pfr=0.28, af=3.0)
    assert v.classify() == "LAG"
