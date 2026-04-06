"""Action-based range updating: narrow villain's range based on observed actions."""

from typing import List

from src.card import Card
from src.board_analysis import BoardTexture
from src.hand_analysis import classify_hand, MadeHandType, DrawType
from src.weighted_range import ComboWeight
from src.sizing_tell import SizingTellInterpreter


# ---------------------------------------------------------------------------
# Tier classification helpers
# ---------------------------------------------------------------------------

def _hand_tier(made: MadeHandType) -> str:
    """Return tier label for a made hand type."""
    if made >= MadeHandType.TRIPS_SET:
        return "monster"
    if made >= MadeHandType.TOP_PAIR_TOP_KICKER:
        return "strong"
    if made >= MadeHandType.MIDDLE_PAIR:
        return "medium"
    if made >= MadeHandType.ACE_HIGH:
        return "weak"
    return "air"


def _draw_tier(draw: DrawType) -> str:
    """Return tier label for a draw type."""
    if draw in (DrawType.COMBO_DRAW_NUT, DrawType.COMBO_DRAW):
        return "draw_strong"
    if draw in (DrawType.FLUSH_DRAW_NUT, DrawType.FLUSH_DRAW_LOW, DrawType.OESD):
        return "draw_medium"
    if draw == DrawType.GUTSHOT:
        return "draw_weak"
    return "no_draw"


def _classify_combo(combo: ComboWeight, board: List[Card]) -> tuple:
    """Return (hand_tier, draw_tier) for a given combo on the board."""
    try:
        hs = classify_hand(combo.combo, board)
        return _hand_tier(hs.made_hand), _draw_tier(hs.draw)
    except Exception:
        return "air", "no_draw"


# ---------------------------------------------------------------------------
# Weight-multiplier tables
# ---------------------------------------------------------------------------

# Multipliers keyed by (hand_tier, draw_tier)
# Each action/sizing category specifies how to scale existing weights.

_CHECK_MULTIPLIERS = {
    # Villain checks: strong hands usually bet → lower strong/monster weight
    "monster": 0.25,
    "strong":  0.45,
    "medium":  1.00,
    "weak":    1.00,
    "air":     0.90,
}

_CHECK_MULTIPLIERS_SLOWPLAY = {
    # OOP on dry board: strong hands check more often (slow-play)
    "monster": 0.55,
    "strong":  0.65,
    "medium":  1.00,
    "weak":    1.00,
    "air":     0.90,
}

_BET_SMALL_MULTIPLIERS = {
    # < 40% pot: merged range (medium value-heavy)
    "monster": 0.70,
    "strong":  0.90,
    "medium":  1.20,
    "weak":    0.80,
    "air":     0.60,
}

_BET_MEDIUM_MULTIPLIERS = {
    # 40-70% pot: standard range
    "monster": 0.85,
    "strong":  1.00,
    "medium":  0.90,
    "weak":    0.50,
    "air":     0.40,
}

_BET_LARGE_MULTIPLIERS = {
    # > 70% pot: polarized
    "monster": 1.00,
    "strong":  0.90,
    "medium":  0.30,
    "weak":    0.15,
    "air":     0.00,   # pure air folds or is a bluff draw
}

_BET_OVERBET_MULTIPLIERS = {
    # > 100% pot: very polarized
    "monster": 1.00,
    "strong":  0.60,
    "medium":  0.10,
    "weak":    0.05,
    "air":     0.00,
}

# Bluff draws keep weight under polarized bets
_DRAW_BLUFF_BONUS = {
    "draw_strong": 0.70,  # used as multiplier in polarized scenarios
    "draw_medium": 0.40,
    "draw_weak":   0.15,
    "no_draw":     0.00,
}

_RAISE_MULTIPLIERS = {
    "monster": 1.00,
    "strong":  0.80,
    "medium":  0.10,
    "weak":    0.00,
    "air":     0.00,
}

_RAISE_DRAW_BONUS = {
    "draw_strong": 0.25,
    "draw_medium": 0.20,
    "draw_weak":   0.05,
    "no_draw":     0.00,
}

_ALL_IN_MULTIPLIERS = {
    "monster": 1.00,
    "strong":  0.70,
    "medium":  0.05,
    "weak":    0.00,
    "air":     0.00,
}

_ALL_IN_DRAW_BONUS = {
    "draw_strong": 0.15,
    "draw_medium": 0.10,
    "draw_weak":   0.00,
    "no_draw":     0.00,
}

_ALL_IN_MONOTONE_FLUSH_MULT = 1.50  # boost flush combos on monotone board

_CALL_MULTIPLIERS = {
    # Calling range: remove nuts (would raise) and pure air (would fold)
    "monster": 0.10,
    "strong":  0.80,
    "medium":  1.00,
    "weak":    0.50,
    "air":     0.00,
}

_CALL_DRAW_KEEP = {
    "draw_strong": 0.85,
    "draw_medium": 0.80,
    "draw_weak":   0.50,
    "no_draw":     1.00,  # no draw → use hand tier multiplier directly
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RangeUpdater:
    """Update villain's weighted range based on observed post-flop actions."""

    def __init__(self) -> None:
        self._tell_interpreter = SizingTellInterpreter()

    def update_range_after_action(
        self,
        current_range: List[ComboWeight],
        board: List[Card],
        action: str,
        sizing_ratio: float,      # bet / pot ratio (0 for check/fold)
        street: str,              # "flop", "turn", "river"
        board_texture: BoardTexture,
        is_ip: bool,              # villain has position?
    ) -> List[ComboWeight]:
        """
        Update villain range weights based on action.

        Actions handled: "check", "bet", "raise", "call", "all_in"
        Returns a new list of ComboWeight with updated weights.
        """
        if not current_range:
            return current_range

        action_lc = action.lower().replace("-", "_")
        is_monotone = getattr(board_texture, 'monotone', False)
        is_dry = getattr(board_texture, 'wetness', 5) <= 3

        # Classify each combo and compute multipliers
        updated: List[ComboWeight] = []

        for cw in current_range:
            hand_tier, draw_tier = _classify_combo(cw, board)
            new_weight = self._compute_weight(
                cw.weight, action_lc, sizing_ratio,
                hand_tier, draw_tier, is_ip, is_dry, is_monotone,
            )
            if new_weight > 0.0:
                updated.append(ComboWeight(combo=cw.combo, weight=new_weight))

        # Renormalize to keep scale manageable (preserve relative weights)
        return self._normalize(updated, current_range)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weight(
        self,
        old_weight: float,
        action: str,
        sizing_ratio: float,
        hand_tier: str,
        draw_tier: str,
        is_ip: bool,
        is_dry: bool,
        is_monotone: bool,
    ) -> float:
        if action == "check":
            if not is_ip and is_dry:
                mult = _CHECK_MULTIPLIERS_SLOWPLAY.get(hand_tier, 1.0)
            else:
                mult = _CHECK_MULTIPLIERS.get(hand_tier, 1.0)
            return old_weight * mult

        if action == "bet":
            if sizing_ratio < 0.40:
                mult = _BET_SMALL_MULTIPLIERS.get(hand_tier, 1.0)
            elif sizing_ratio <= 0.70:
                mult = _BET_MEDIUM_MULTIPLIERS.get(hand_tier, 1.0)
            elif sizing_ratio <= 1.00:
                mult = _BET_LARGE_MULTIPLIERS.get(hand_tier, 0.0)
                # Bluff draws keep some weight in polarized scenarios
                if mult == 0.0 and draw_tier != "no_draw":
                    mult = _DRAW_BLUFF_BONUS.get(draw_tier, 0.0)
            else:
                mult = _BET_OVERBET_MULTIPLIERS.get(hand_tier, 0.0)
                if mult == 0.0 and draw_tier != "no_draw":
                    mult = _DRAW_BLUFF_BONUS.get(draw_tier, 0.0)
            return old_weight * mult

        if action == "raise":
            mult = _RAISE_MULTIPLIERS.get(hand_tier, 0.0)
            if mult == 0.0 and draw_tier != "no_draw":
                mult = _RAISE_DRAW_BONUS.get(draw_tier, 0.0)
            return old_weight * mult

        if action == "all_in":
            mult = _ALL_IN_MULTIPLIERS.get(hand_tier, 0.0)
            if mult == 0.0 and draw_tier != "no_draw":
                mult = _ALL_IN_DRAW_BONUS.get(draw_tier, 0.0)
            # Monotone board: flush combos get extra weight
            if is_monotone and mult > 0:
                # detect flush combos via draw_tier
                if draw_tier in ("draw_strong", "draw_medium"):
                    mult = min(1.0, mult * _ALL_IN_MONOTONE_FLUSH_MULT)
            return old_weight * mult

        if action == "call":
            # Draws use draw-specific multiplier layered on top of tier mult
            tier_mult = _CALL_MULTIPLIERS.get(hand_tier, 1.0)
            if draw_tier != "no_draw":
                draw_mult = _CALL_DRAW_KEEP.get(draw_tier, 1.0)
                return old_weight * max(tier_mult, draw_mult)
            return old_weight * tier_mult

        # Unknown action: no update
        return old_weight

    @staticmethod
    def _normalize(
        updated: List[ComboWeight],
        original: List[ComboWeight],
    ) -> List[ComboWeight]:
        """Rescale so mean weight stays close to original mean weight."""
        if not updated or not original:
            return updated
        orig_mean = sum(cw.weight for cw in original) / len(original)
        new_total = sum(cw.weight for cw in updated)
        if new_total == 0:
            return updated
        new_mean = new_total / len(updated)
        if new_mean == 0:
            return updated
        scale = orig_mean / new_mean
        return [
            ComboWeight(combo=cw.combo, weight=round(cw.weight * scale, 6))
            for cw in updated
        ]
