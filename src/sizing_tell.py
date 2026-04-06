"""Sizing-based range interpretation: infer opponent range from bet sizing."""

from dataclasses import dataclass

from src.board_analysis import BoardTexture
from src.opponent import VillainProfile


@dataclass
class SizingInterpretation:
    polarization: str           # "merged", "standard", "polarized", "very_polarized"
    estimated_value_pct: float  # estimated fraction of value hands in villain's range
    estimated_bluff_pct: float  # estimated fraction of bluffs in villain's range
    range_strength: str         # "strong", "medium", "weak_or_polar"
    description: str            # human-readable explanation


class SizingTellInterpreter:
    """Interpret opponent bet sizing to infer their likely range composition."""

    def interpret(
        self,
        action: str,                   # "bet", "raise", "all_in"
        sizing_ratio: float,           # bet / pot ratio
        street: str,                   # "flop", "turn", "river"
        board_texture: BoardTexture,
        villain_profile: VillainProfile = None,
    ) -> SizingInterpretation:
        """
        Interpret opponent sizing to estimate range polarization.

        Core rules:
        - Small bet (< 35% pot): merged range, medium strength
        - Medium bet (35-65% pot): standard range, value + bluffs
        - Large bet (65-100% pot): polarized, strong + bluffs, few medium
        - Overbet (> 100% pot): very polarized, nuts or pure bluffs
        - All-in: depends on SPR and board texture
        """
        action_lc = action.lower()
        is_river = street.lower() == "river"
        is_monotone = getattr(board_texture, 'monotone', False)

        if action_lc == "all_in":
            return self._interpret_all_in(board_texture, is_monotone)

        # Base interpretation from sizing
        result = self._base_from_sizing(sizing_ratio, is_river)

        # Board-texture adjustments
        result = self._adjust_for_board(result, sizing_ratio, board_texture, is_monotone, is_river)

        # Adjust using villain profile if available
        if villain_profile is not None and villain_profile.stats.hands_played >= 100:
            result = self._adjust_for_profile(result, sizing_ratio, villain_profile)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_from_sizing(self, ratio: float, is_river: bool) -> SizingInterpretation:
        """Return base interpretation purely from sizing ratio."""
        if ratio < 0.35:
            return SizingInterpretation(
                polarization="merged",
                estimated_value_pct=0.55,
                estimated_bluff_pct=0.12,
                range_strength="medium",
                description=(
                    f"Small bet ({ratio:.0%} pot): merged range, "
                    "medium-strength value, few bluffs"
                ),
            )
        if ratio <= 0.65:
            return SizingInterpretation(
                polarization="standard",
                estimated_value_pct=0.60,
                estimated_bluff_pct=0.25,
                range_strength="medium",
                description=(
                    f"Medium bet ({ratio:.0%} pot): balanced range, "
                    "value + standard bluff frequency"
                ),
            )
        if ratio <= 1.00:
            base_bluff = 0.35 if is_river else 0.30
            return SizingInterpretation(
                polarization="polarized",
                estimated_value_pct=0.55,
                estimated_bluff_pct=base_bluff,
                range_strength="weak_or_polar",
                description=(
                    f"Large bet ({ratio:.0%} pot): polarized range, "
                    "strong value or bluffs, few middle-strength hands"
                ),
            )
        # Overbet
        base_bluff = 0.40 if is_river else 0.33
        return SizingInterpretation(
            polarization="very_polarized",
            estimated_value_pct=0.50,
            estimated_bluff_pct=base_bluff,
            range_strength="weak_or_polar",
            description=(
                f"Overbet ({ratio:.0%} pot): very polarized – "
                "near-nut value or pure bluffs"
            ),
        )

    def _interpret_all_in(
        self, board_texture: BoardTexture, is_monotone: bool
    ) -> SizingInterpretation:
        """Interpret an all-in action."""
        if is_monotone:
            # Monotone board + all-in: bluffing is very risky for villain,
            # so value-heavy
            return SizingInterpretation(
                polarization="very_polarized",
                estimated_value_pct=0.75,
                estimated_bluff_pct=0.12,
                range_strength="strong",
                description=(
                    "All-in on monotone board: extremely value-heavy "
                    "(villain unlikely to bluff-shove monotone board)"
                ),
            )
        # Standard all-in
        wetness = getattr(board_texture, 'wetness', 5)
        if wetness >= 7:
            # Wet board + all-in: still value-heavy but some draws possible
            return SizingInterpretation(
                polarization="very_polarized",
                estimated_value_pct=0.65,
                estimated_bluff_pct=0.18,
                range_strength="strong",
                description=(
                    "All-in on wet board: value-heavy, "
                    "some semi-bluff draws possible"
                ),
            )
        return SizingInterpretation(
            polarization="very_polarized",
            estimated_value_pct=0.60,
            estimated_bluff_pct=0.15,
            range_strength="strong",
            description="All-in: polarized – strong value or committed semi-bluff",
        )

    def _adjust_for_board(
        self,
        base: SizingInterpretation,
        ratio: float,
        board_texture: BoardTexture,
        is_monotone: bool,
        is_river: bool,
    ) -> SizingInterpretation:
        """Apply board-texture overrides."""
        value_pct = base.estimated_value_pct
        bluff_pct = base.estimated_bluff_pct
        desc = base.description
        polarization = base.polarization
        strength = base.range_strength

        wetness = getattr(board_texture, 'wetness', 5)

        if is_monotone and ratio > 0.65:
            # Large bet on monotone board → very value-heavy
            value_pct = min(0.80, value_pct * 1.20)
            bluff_pct = max(0.10, bluff_pct * 0.60)
            desc += "; monotone board makes bluffing risky → more value"
            strength = "strong"

        elif is_river and ratio > 1.00:
            # River overbet → very polarized
            polarization = "very_polarized"
            desc += "; river overbet is extremely polarized"

        elif wetness >= 7 and ratio > 0.65:
            # Wet board + large bet → likely nut draw or strong made hand
            value_pct = min(0.75, value_pct * 1.10)
            desc += "; wet board + large sizing suggests strong hand or nut draw"

        elif wetness <= 3 and ratio < 0.35:
            # Dry board + small bet → range cbet, could be wide/weak
            bluff_pct = min(0.30, bluff_pct * 1.50)
            desc += "; dry board small bet may be a wide range cbet"

        return SizingInterpretation(
            polarization=polarization,
            estimated_value_pct=round(value_pct, 3),
            estimated_bluff_pct=round(bluff_pct, 3),
            range_strength=strength,
            description=desc,
        )

    def _adjust_for_profile(
        self,
        base: SizingInterpretation,
        ratio: float,
        villain_profile: VillainProfile,
    ) -> SizingInterpretation:
        """Adjust interpretation based on observed villain tendencies."""
        stats = villain_profile.stats
        value_pct = base.estimated_value_pct
        bluff_pct = base.estimated_bluff_pct
        desc = base.description

        # Determine villain's sizing tendency from historical data
        avg_sizing = (
            stats.avg_bet_sizing.get("flop_cbet", 0.55)
            + stats.avg_bet_sizing.get("turn_bet", 0.65)
            + stats.avg_bet_sizing.get("river_bet", 0.70)
        ) / 3

        sizing_tendency = "small" if avg_sizing < 0.45 else ("large" if avg_sizing > 0.70 else "normal")

        if sizing_tendency == "small" and ratio < 0.35:
            # Villain usually bets small → small bet doesn't indicate weakness
            desc += f"; villain typically bets small ({avg_sizing:.0%} avg), small bet is his norm"
        elif sizing_tendency == "large" and ratio > 0.70:
            # Villain usually bets large → large bet doesn't indicate polarization
            bluff_pct = max(0.15, bluff_pct * 0.80)
            desc += f"; villain typically bets large ({avg_sizing:.0%} avg), large bet less polarizing"

        # High aggression factor → more bluffs in range
        if stats.aggression_factor > 3.0:
            bluff_pct = min(0.45, bluff_pct * 1.20)
            desc += f"; LAG villain (AF={stats.aggression_factor:.1f}) → bluff frequency higher"
        elif stats.aggression_factor < 1.5:
            bluff_pct = max(0.05, bluff_pct * 0.70)
            value_pct = min(0.90, value_pct * 1.10)
            desc += f"; passive villain (AF={stats.aggression_factor:.1f}) → mostly value"

        return SizingInterpretation(
            polarization=base.polarization,
            estimated_value_pct=round(value_pct, 3),
            estimated_bluff_pct=round(bluff_pct, 3),
            range_strength=base.range_strength,
            description=desc,
        )
