"""Villain stat tracking and exploitation profiling."""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from src.position import Position


@dataclass
class VillainStats:
    hands_played: int = 0
    vpip: float = 0.25
    pfr: float = 0.20
    rfi_by_position: Dict[Position, float] = field(
        default_factory=lambda: {p: 0.0 for p in Position}
    )
    fold_to_3bet_by_position: Dict[Position, float] = field(
        default_factory=lambda: {p: 0.65 for p in Position}
    )
    cbet_flop: float = 0.60
    cbet_turn: float = 0.50
    cbet_river: float = 0.40
    fold_to_flop_cbet: float = 0.55
    fold_to_turn_cbet: float = 0.50
    fold_to_river_cbet: float = 0.45
    check_raise_flop: float = 0.08
    probe_bet: float = 0.30
    float_flop: float = 0.25
    aggression_factor: float = 2.0
    wtsd: float = 0.28
    wsd: float = 0.52
    avg_bet_sizing: Dict[str, float] = field(
        default_factory=lambda: {
            "flop_cbet": 0.55,
            "turn_bet": 0.65,
            "river_bet": 0.70,
        }
    )
    # Raw counters for Bayesian updating
    _vpip_hands: int = field(default=0, repr=False)
    _vpip_count: int = field(default=0, repr=False)
    _pfr_hands: int = field(default=0, repr=False)
    _pfr_count: int = field(default=0, repr=False)
    _cbet_opps: int = field(default=0, repr=False)
    _cbet_count: int = field(default=0, repr=False)
    _fold_cbet_opps: int = field(default=0, repr=False)
    _fold_cbet_count: int = field(default=0, repr=False)


class VillainProfile:
    """Track and exploit a villain's tendencies."""

    def __init__(self) -> None:
        self.stats = VillainStats()

    # ------------------------------------------------------------------
    # Stat updates
    # ------------------------------------------------------------------

    def update_action(
        self,
        street: str,
        action: str,
        position: Position,
        amount: float,
        pot: float,
    ) -> None:
        """Update stats from an observed villain action."""
        s = self.stats
        s.hands_played += 1
        action = action.lower()

        if street == "preflop":
            s._vpip_hands += 1
            if action in ("call", "raise", "3bet", "4bet", "limp"):
                s._vpip_count += 1
            if action in ("raise", "3bet", "4bet"):
                s._pfr_count += 1
            s._pfr_hands += 1
            # Update VPIP/PFR with smoothing
            prior_weight = min(s.hands_played, 100)
            s.vpip = (s._vpip_count + 0.25 * 20) / (s._vpip_hands + 20)
            s.pfr = (s._pfr_count + 0.20 * 20) / (s._pfr_hands + 20)

            # Update RFI
            if action in ("raise",):
                prev = s.rfi_by_position.get(position, 0.0)
                s.rfi_by_position[position] = prev * 0.9 + 0.1 * 1.0
            else:
                prev = s.rfi_by_position.get(position, 0.0)
                s.rfi_by_position[position] = prev * 0.9

            # Update fold-to-3bet
            if action == "fold":
                prev = s.fold_to_3bet_by_position.get(position, 0.65)
                s.fold_to_3bet_by_position[position] = min(0.99, prev * 0.9 + 0.1 * 1.0)
            elif action in ("call", "raise", "4bet"):
                prev = s.fold_to_3bet_by_position.get(position, 0.65)
                s.fold_to_3bet_by_position[position] = max(0.01, prev * 0.9)

        elif street == "flop":
            if action == "bet":
                s._cbet_count += 1
                s._cbet_opps += 1
                if pot > 0:
                    sizing_key = "flop_cbet"
                    old = s.avg_bet_sizing.get(sizing_key, 0.55)
                    s.avg_bet_sizing[sizing_key] = old * 0.8 + 0.2 * (amount / pot)
            elif action == "check":
                s._cbet_opps += 1
            elif action == "fold":
                s._fold_cbet_count += 1
                s._fold_cbet_opps += 1
            elif action in ("call", "raise"):
                s._fold_cbet_opps += 1

            if s._cbet_opps > 0:
                s.cbet_flop = (s._cbet_count + 0.60 * 10) / (s._cbet_opps + 10)
            if s._fold_cbet_opps > 0:
                s.fold_to_flop_cbet = (s._fold_cbet_count + 0.55 * 10) / (s._fold_cbet_opps + 10)

        elif street == "turn":
            if action == "bet" and pot > 0:
                old = s.avg_bet_sizing.get("turn_bet", 0.65)
                s.avg_bet_sizing["turn_bet"] = old * 0.8 + 0.2 * (amount / pot)
            # Simplified turn updates
            if action == "bet":
                s.cbet_turn = s.cbet_turn * 0.9 + 0.1 * 1.0
            elif action == "check":
                s.cbet_turn = s.cbet_turn * 0.9
            if action == "fold":
                s.fold_to_turn_cbet = min(0.99, s.fold_to_turn_cbet * 0.9 + 0.1)
            elif action in ("call", "raise"):
                s.fold_to_turn_cbet = max(0.01, s.fold_to_turn_cbet * 0.9)

        elif street == "river":
            if action == "bet" and pot > 0:
                old = s.avg_bet_sizing.get("river_bet", 0.70)
                s.avg_bet_sizing["river_bet"] = old * 0.8 + 0.2 * (amount / pot)
            if action == "fold":
                s.fold_to_river_cbet = min(0.99, s.fold_to_river_cbet * 0.9 + 0.1)
            elif action in ("call", "raise"):
                s.fold_to_river_cbet = max(0.01, s.fold_to_river_cbet * 0.9)

        # Update aggression factor
        if action in ("bet", "raise"):
            s.aggression_factor = min(10.0, s.aggression_factor * 0.95 + 0.5)
        elif action == "call":
            s.aggression_factor = max(0.1, s.aggression_factor * 0.98)

    # ------------------------------------------------------------------
    # Exploitation
    # ------------------------------------------------------------------

    def get_exploit_suggestion(self, street: str, situation: str) -> str:
        """Return human-readable exploit advice for a given situation."""
        s = self.stats
        suggestions: List[str] = []

        if situation == "facing_cbet":
            if s.cbet_flop > 0.75:
                suggestions.append("Villain cbets very frequently – float or raise with good hands")
            if s.fold_to_flop_cbet < 0.40:
                suggestions.append("Villain rarely folds to cbets – don't bluff, value-bet thinly")
            elif s.fold_to_flop_cbet > 0.65:
                suggestions.append("Villain folds a lot – consider raising as bluff")

        elif situation == "cbet_opportunity":
            if s.fold_to_flop_cbet > 0.65:
                suggestions.append("Villain folds to cbets frequently – bet wide for profit")
            if s.check_raise_flop > 0.15:
                suggestions.append("Villain check-raises often – be careful with thin cbets")

        elif situation == "facing_3bet":
            avg_fold = sum(s.fold_to_3bet_by_position.values()) / max(
                len(s.fold_to_3bet_by_position), 1
            )
            if avg_fold > 0.75:
                suggestions.append("Villain folds to 3bets too often – 3bet bluff liberally")
            elif avg_fold < 0.40:
                suggestions.append("Villain calls 3bets wide – only 3bet for value")

        elif situation == "steal_blinds":
            avg_fold_3b = sum(s.fold_to_3bet_by_position.values()) / max(
                len(s.fold_to_3bet_by_position), 1
            )
            if s.vpip < 0.15:
                suggestions.append("Nit in blind – open very wide, steal often")
            elif avg_fold_3b > 0.70:
                suggestions.append("Villain defends blinds passively – open wide and cbet")

        elif situation == "probe_opportunity":
            if s.probe_bet > 0.50:
                suggestions.append("Villain probes often when checked to – check back strong hands to induce")
            if s.cbet_turn < 0.30:
                suggestions.append("Villain rarely bets turn – probe bet for value and bluffs")

        # Player type summary
        if self.is_fish():
            suggestions.append("Fish detected: value-bet aggressively, minimise bluffs")
        elif self.is_nit():
            suggestions.append("Nit detected: respect raises, steal blinds frequently")
        elif self.is_lag():
            suggestions.append("LAG detected: tighten up, trap with strong hands, x/r bluffs")
        elif self.is_tag():
            suggestions.append("TAG detected: play straightforwardly, avoid fancy plays")

        return "; ".join(suggestions) if suggestions else "No specific exploit identified"

    def estimate_range(
        self, street: str, action_line: List[str], position: Position
    ) -> Set[str]:
        """Heuristically estimate villain's range from action line."""
        from src.ranges import (
            get_rfi_range, get_3bet_range, get_4bet_range,
            get_call_rfi_range, get_call_3bet_range,
        )
        action_line_lower = [a.lower() for a in action_line]

        # Preflop range estimation
        if "4bet" in action_line_lower:
            return get_4bet_range(position)
        if "3bet" in action_line_lower:
            return get_3bet_range(position, position)  # rough estimate
        if "raise" in action_line_lower or "open" in action_line_lower:
            return get_rfi_range(position)
        if "call" in action_line_lower:
            # Called a raise
            return get_call_rfi_range(position, position)

        # Default
        return get_rfi_range(position)

    # ------------------------------------------------------------------
    # Player-type classification
    # ------------------------------------------------------------------

    def is_fish(self) -> bool:
        """VPIP > 40%, PFR < 15%."""
        return self.stats.vpip > 0.40 and self.stats.pfr < 0.15

    def is_nit(self) -> bool:
        """VPIP < 15%."""
        return self.stats.vpip < 0.15

    def is_lag(self) -> bool:
        """VPIP > 30%, PFR > 25%, AF > 3."""
        return (
            self.stats.vpip > 0.30
            and self.stats.pfr > 0.25
            and self.stats.aggression_factor > 3.0
        )

    def is_tag(self) -> bool:
        """VPIP 20-30%, PFR 15-25%."""
        return (
            0.20 <= self.stats.vpip <= 0.30
            and 0.15 <= self.stats.pfr <= 0.25
        )
