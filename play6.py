#!/usr/bin/env python3
"""
play6.py — 6-Max NLHE 6人桌对战模式 (1人 vs 5 Bot)
运行方式: python play6.py
"""

from __future__ import annotations

import sys
import os
import random
from typing import Dict, List, Optional, Tuple

# 确保能找到 src/ 目录
sys.path.insert(0, os.path.dirname(__file__))

from src.card import Card, Rank, Suit
from src.position import Position
from src.bot import PokerBot, GameState
from src.evaluator import evaluate_7
from src.hand_analysis import classify_hand
from src.game_engine import (
    GameDeck,
    card_display,
    cards_display,
    hand_rank_description,
    calc_min_raise,
)
from src.game_engine_6max import (
    PlayerState6,
    SidePot,
    HandResult6,
    SIX_MAX_POSITIONS,
    POSITION_NAMES,
    PREFLOP_ORDER,
    POSTFLOP_ORDER,
    SB_AMOUNT,
    BB_AMOUNT,
    calculate_side_pots,
    betting_round_6max,
)

# ---------------------------------------------------------------------------
# 全局游戏配置
# ---------------------------------------------------------------------------
STARTING_STACK = 100.0   # BB

# 玩家在桌子上的固定座位索引（人类玩家固定在 seat 3 = BTN）
HUMAN_SEAT = 3  # BTN 位置

# Bot 名字
BOT_NAMES = ["Bot-1", "Bot-2", "Bot-3", "Bot-4", "Bot-5"]

# ---------------------------------------------------------------------------
# 显示工具
# ---------------------------------------------------------------------------

def banner() -> None:
    print("╔══════════════════════════════════════╗")
    print("║     6-Max NLHE Poker Bot v2          ║")
    print("║     6人桌对战模式 (1v5)              ║")
    print("╚══════════════════════════════════════╝")
    print()


def show_help() -> None:
    print("\n─── 帮助 ───────────────────────────────")
    print("  f / fold          弃牌")
    print("  c / call          跟注")
    print("  ch / check        过牌（当 to_call=0 时）")
    print("  r <数字>          加注到 <数字>BB，例如: r 6")
    print("  raise <数字>      同上")
    print("  b <数字>          下注 <数字>BB（翻后无人下注时）")
    print("  bet <数字>        同上")
    print("  a / allin         全下")
    print("  q                 退出")
    print("  h                 显示帮助")
    print("  s                 显示统计")
    print("  table             显示当前座位表")
    print("  thinking          切换 Bot 思考显示")
    print("────────────────────────────────────────\n")


def show_table(players: List[PlayerState6], btn_seat: int) -> None:
    """显示座位表。"""
    print("\n座位表:")
    for i, p in enumerate(players):
        marker = "  ← 你在这里" if p.is_human else ""
        sb_bb = ""
        if p.position == Position.SB:
            sb_bb = f" (SB {SB_AMOUNT}BB)"
        elif p.position == Position.BB:
            sb_bb = f" (BB {BB_AMOUNT}BB)"
        print(f"  {p.pos_name:<4}: {p.name:<8} [{p.stack:.1f}BB]{sb_bb}{marker}")
    print()


def show_stats_6max(
    hand_num: int,
    players: List[PlayerState6],
    wins: Dict[str, int],
    starting_stack: float,
    human_name: str,
) -> None:
    """显示全局统计。"""
    print()
    print("╔═══════════════════════════════════════════╗")
    print("║              对战统计                      ║")
    print("╠═══════════════════════════════════════════╣")
    print(f"║  总手数: {hand_num:<34}║")
    print("║                                           ║")
    for p in players:
        w = wins.get(p.name, 0)
        profit = p.stack - starting_stack
        sign = "+" if profit >= 0 else ""
        line = f"  {p.name:<8} 赢 {w:<4} 手  筹码: {p.stack:<7.1f}BB {sign}{profit:.1f}BB"
        print(f"║{line:<43}║")
    print("║                                           ║")
    # 计算人类玩家的 bb/100
    human_stack = next((p.stack for p in players if p.is_human), starting_stack)
    human_profit = human_stack - starting_stack
    bb100 = int(human_profit / hand_num * 100) if hand_num > 0 else 0
    print(f"║  你的 bb/100: {bb100:+d}{' ' * max(0, 29 - len(f'{bb100:+d}'))}║")
    print("╚═══════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# 用户输入解析
# ---------------------------------------------------------------------------

def parse_action(
    raw: str,
    to_call: float,
    player_stack: float,
    min_raise: float,
) -> tuple:
    """
    解析用户输入，返回 (action, amount) 或 (None, None)。
    特殊返回: ("quit",0), ("help",0), ("status",0), ("toggle_thinking",0), ("table",0)
    """
    tokens = raw.strip().lower().split()
    if not tokens:
        return None, None

    cmd = tokens[0]

    if cmd in ("q", "quit"):
        return "quit", 0.0
    if cmd in ("h", "help"):
        return "help", 0.0
    if cmd in ("s", "status"):
        return "status", 0.0
    if cmd == "thinking":
        return "toggle_thinking", 0.0
    if cmd == "table":
        return "table", 0.0
    if cmd in ("f", "fold"):
        return "fold", 0.0
    if cmd in ("c", "call"):
        actual_call = min(to_call, player_stack)
        return "call", actual_call
    if cmd in ("ch", "check"):
        if to_call > 0:
            print(f"  ⚠ 不能过牌，你需要跟注 {to_call:.1f}BB 或弃牌。")
            return None, None
        return "check", 0.0
    if cmd in ("a", "all", "allin"):
        return "allin", player_stack
    if cmd in ("r", "raise", "b", "bet"):
        if len(tokens) < 2:
            print(f"  ⚠ 请指定金额，例如: r 6 (表示加注到 6BB)")
            return None, None
        try:
            amount = float(tokens[1])
        except ValueError:
            print(f"  ⚠ 无效金额: {tokens[1]}")
            return None, None
        if amount >= player_stack:
            return "allin", player_stack
        if amount < min_raise - 0.001:
            print(f"  ⚠ 最小加注金额为 {min_raise:.1f}BB")
            return None, None
        return "raise", amount

    print(f"  ⚠ 未知命令: {raw}  (输入 h 查看帮助)")
    return None, None


# ---------------------------------------------------------------------------
# Bot 思考格式化
# ---------------------------------------------------------------------------

def format_bot_thinking(
    decision: dict,
    bot_name: str,
    bot_position: Position,
    bot_hand: tuple,
    board: list,
) -> str:
    lines = [f"{bot_name} ({POSITION_NAMES[bot_position]}) 思考中..."]

    if board and "hand_strength" in decision and decision["hand_strength"] is not None:
        hs = decision["hand_strength"]
        lines.append(f"  手牌分类: {hs.made_hand.name}")
        if hs.draw.name != "NONE":
            lines.append(f"  听牌: {hs.draw.name}")

    if "barrel_plan" in decision and decision["barrel_plan"] is not None:
        bp = decision["barrel_plan"]
        lines.append(f"  Barrel 计划: {bp.flop_action}/{bp.turn_action}/{bp.river_action}")

    action = decision.get("action", "?")
    amount = decision.get("amount", 0.0)
    if action in ("raise", "bet", "3bet", "4bet") and amount > 0:
        lines.append(f"  决定: {action.upper()} {amount:.1f}BB")
    elif action in ("allin", "all-in"):
        lines.append(f"  决定: ALL-IN")
    else:
        lines.append(f"  决定: {action.upper()}")

    reasoning = decision.get("reasoning", "")
    if reasoning:
        lines.append(f"  理由: \"{reasoning}\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6人桌单手牌引擎
# ---------------------------------------------------------------------------

class HandEngine6Max:
    """管理一手 6-max 牌的完整流程。"""

    def __init__(
        self,
        players: List[PlayerState6],
        bots: Dict[str, PokerBot],
        show_thinking: bool,
        hand_num: int,
    ) -> None:
        self.players = players
        self.bots = bots
        self.show_thinking = show_thinking
        self.hand_num = hand_num
        self.board: List[Card] = []
        self.pot: float = 0.0
        self.action_history: list = []
        self.last_raise_increment: float = BB_AMOUNT

        # 重置每个玩家本手状态
        for p in players:
            p.bet_street = 0.0
            p.total_invested = 0.0
            p.folded = False
            p.is_allin = False
            p.hole_cards = []

        # 发牌
        deck = GameDeck()
        for p in players:
            p.hole_cards = deck.deal_n(2)
        self.deck = deck

        # 找到人类玩家
        self.human_idx: int = next(i for i, p in enumerate(players) if p.is_human)

    # ── 辅助 ──────────────────────────────────────────────────────────────

    def _active_players(self) -> List[int]:
        """返回未弃牌的玩家索引。"""
        return [i for i, p in enumerate(self.players) if not p.folded]

    def _pos_order(self, base_order: List[Position]) -> List[int]:
        """按给定位置顺序返回玩家索引列表（跳过不在桌的位置）。"""
        pos_to_idx = {p.position: i for i, p in enumerate(self.players)}
        result = []
        for pos in base_order:
            if pos in pos_to_idx:
                result.append(pos_to_idx[pos])
        return result

    def _show_board(self, street: str) -> None:
        street_names = {"flop": "翻牌", "turn": "转牌", "river": "河牌"}
        active = [self.players[i].name for i in self._active_players()]
        print(f"\n--- {street_names.get(street, street)} ---")
        print(f"牌面: {cards_display(self.board)}")
        print(f"底池: {self.pot:.1f}BB")
        if len(active) <= 4:
            print(f"还剩 {len(active)} 人: {', '.join(active)}")

    def _show_hand_header(self) -> None:
        print(f"\n{'━' * 10} 第 {self.hand_num} 手 {'━' * 10}")
        print("座位表:")
        for p in self.players:
            marker = "  ← 你的位置" if p.is_human else ""
            sb_bb = ""
            if p.position == Position.SB:
                sb_bb = f" (SB {SB_AMOUNT}BB)"
            elif p.position == Position.BB:
                sb_bb = f" (BB {BB_AMOUNT}BB)"
            print(f"  {p.pos_name:<4}: {p.name:<8} [{p.stack:.1f}BB]{sb_bb}{marker}")
        print()

    # ── 下注轮 ────────────────────────────────────────────────────────────

    def _do_betting_round(self, street: str) -> bool:
        """
        执行一轮下注。
        返回 True 继续游戏，False 表示手牌结束（全弃牌）。
        """
        if street == "preflop":
            order = self._pos_order(PREFLOP_ORDER)
        else:
            order = self._pos_order(POSTFLOP_ORDER)

        # 过滤掉弃牌/allin 的玩家（他们不需要行动，但仍在 order 中以保持顺序）
        pot, lri, continues = betting_round_6max(
            street=street,
            players=self.players,
            action_order=order,
            pot=self.pot,
            action_history=self.action_history,
            last_raise_increment=self.last_raise_increment,
            on_player_action=self._handle_action,
        )
        self.pot = pot
        self.last_raise_increment = lri
        return continues

    def _handle_action(
        self,
        player_idx: int,
        to_call: float,
        pot: float,
        min_raise: float,
        last_raise_increment: float,
    ) -> Tuple[str, float, float, float]:
        """
        处理单个玩家的行动。
        返回 (action, amount, new_pot, new_last_raise_increment)
        """
        p = self.players[player_idx]

        if p.is_human:
            action, amount, new_pot, new_lri = self._human_action(
                player_idx, to_call, pot, min_raise, last_raise_increment
            )
        else:
            action, amount, new_pot, new_lri = self._bot_action(
                player_idx, to_call, pot, min_raise, last_raise_increment
            )

        return action, amount, new_pot, new_lri

    def _human_action(
        self,
        player_idx: int,
        to_call: float,
        pot: float,
        min_raise: float,
        last_raise_increment: float,
    ) -> Tuple[str, float, float, float]:
        """提示人类玩家输入，返回 (action, amount, new_pot, new_lri)。"""
        p = self.players[player_idx]
        human_hand = cards_display(p.hole_cards)

        # 构建提示
        options = []
        if to_call > 0:
            options.append("[f]old")
            actual_call = min(to_call, p.stack)
            options.append(f"[c]all {actual_call:.1f}BB")
        else:
            options.append("[ch]eck")

        if p.stack > to_call:
            if to_call > 0:
                options.append(f"[r]aise 到多少BB (最小 {min_raise:.1f}BB)")
            else:
                options.append("[b]et 多少BB")
        options.append(f"[a]llin ({p.stack + p.bet_street:.1f}BB total)")

        prompt = " / ".join(options)

        while True:
            try:
                raw = input(f"\n> 你的选择 ({p.pos_name}) {prompt}? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出游戏。")
                sys.exit(0)

            action, amount = parse_action(raw, to_call, p.stack, min_raise)

            if action == "quit":
                print("\n退出游戏。")
                sys.exit(0)
            elif action == "help":
                show_help()
                continue
            elif action == "status":
                print(f"\n  底池: {pot:.1f}BB | 你: {p.stack:.1f}BB")
                print(f"  牌面: {cards_display(self.board) if self.board else '(翻前)'}")
                print(f"  你的手牌: {human_hand}")
                continue
            elif action == "toggle_thinking":
                self.show_thinking = not self.show_thinking
                state = "开启" if self.show_thinking else "关闭"
                print(f"  Bot 思考过程已{state}。")
                continue
            elif action == "table":
                show_table(self.players, -1)
                continue
            elif action is None:
                continue

            # 执行行动
            if action == "fold":
                print(f"  你弃牌。")
                self.action_history.append({
                    "player": "hero", "actor": 0, "action": "fold",
                    "amount": 0.0, "position": p.position, "street": self._current_street,
                    "pot": pot,
                })
                return "fold", 0.0, pot, last_raise_increment

            elif action == "check":
                print(f"  你过牌。")
                self.action_history.append({
                    "player": "hero", "actor": 0, "action": "check",
                    "amount": 0.0, "position": p.position, "street": self._current_street,
                    "pot": pot,
                })
                return "check", 0.0, pot, last_raise_increment

            elif action == "call":
                actual_call = min(to_call, p.stack)
                p.stack -= actual_call
                p.bet_street += actual_call
                pot += actual_call
                if p.stack <= 0:
                    p.is_allin = True
                    print(f"  你跟注 {actual_call:.1f}BB（全下）。")
                else:
                    print(f"  你跟注 {actual_call:.1f}BB。")
                self.action_history.append({
                    "player": "hero", "actor": 0, "action": "call",
                    "amount": actual_call, "position": p.position, "street": self._current_street,
                    "pot": pot,
                })
                return "call", actual_call, pot, last_raise_increment

            elif action in ("raise", "allin"):
                if action == "allin":
                    raise_to = p.bet_street + p.stack
                else:
                    raise_to = amount

                additional = raise_to - p.bet_street
                additional = min(additional, p.stack)
                p.stack -= additional
                old_bet = p.bet_street
                p.bet_street += additional
                pot += additional

                new_lri = p.bet_street - old_bet if p.bet_street > old_bet else last_raise_increment

                if p.stack <= 0:
                    p.is_allin = True
                    print(f"  你全下: {p.bet_street:.1f}BB。")
                    self.action_history.append({
                        "player": "hero", "actor": 0, "action": "all-in",
                        "amount": p.bet_street, "position": p.position, "street": self._current_street,
                        "pot": pot,
                    })
                    return "allin", p.bet_street, pot, new_lri
                else:
                    print(f"  你加注到 {p.bet_street:.1f}BB。")
                    self.action_history.append({
                        "player": "hero", "actor": 0, "action": "raise",
                        "amount": p.bet_street, "position": p.position, "street": self._current_street,
                        "pot": pot,
                    })
                    return "raise", p.bet_street, pot, new_lri

    def _bot_action(
        self,
        player_idx: int,
        to_call: float,
        pot: float,
        min_raise: float,
        last_raise_increment: float,
    ) -> Tuple[str, float, float, float]:
        """Bot 决策并执行，返回 (action, amount, new_pot, new_lri)。"""
        p = self.players[player_idx]
        bot = self.bots[p.name]

        # 构建 GameState
        active = [i for i in range(len(self.players)) if not self.players[i].folded and i != player_idx]
        villain_positions = [self.players[i].position for i in active]
        villain_stacks = [self.players[i].stack for i in active]
        num_active = len([pp for pp in self.players if not pp.folded])

        gs = GameState(
            hero_hand=tuple(p.hole_cards),
            board=list(self.board),
            pot=pot,
            to_call=to_call,
            hero_stack=p.stack,
            villain_stacks=villain_stacks,
            hero_position=p.position,
            villain_positions=villain_positions,
            street=self._current_street,
            action_history=self.action_history,
            is_tournament=False,
            num_players=num_active,
        )

        decision = bot.decide(gs)
        action = decision.get("action", "fold")
        amount = decision.get("amount", 0.0)

        if self.show_thinking:
            print(format_bot_thinking(decision, p.name, p.position, tuple(p.hole_cards), self.board))

        pos_name = p.pos_name

        if action == "fold":
            reasoning = decision.get("reasoning", "")
            print(f"\n{p.name} ({pos_name}): FOLD")
            if reasoning and self.show_thinking:
                print(f'  理由: "{reasoning}"')
            self.action_history.append({
                "player": "villain", "actor": player_idx, "action": "fold",
                "amount": 0.0, "position": p.position, "street": self._current_street,
                "pot": pot,
            })
            return "fold", 0.0, pot, last_raise_increment

        elif action == "check":
            print(f"\n{p.name} ({pos_name}): CHECK")
            self.action_history.append({
                "player": "villain", "actor": player_idx, "action": "check",
                "amount": 0.0, "position": p.position, "street": self._current_street,
                "pot": pot,
            })
            return "check", 0.0, pot, last_raise_increment

        elif action == "call":
            actual_call = min(to_call, p.stack)
            p.stack -= actual_call
            p.bet_street += actual_call
            pot += actual_call
            if p.stack <= 0:
                p.is_allin = True
                print(f"\n{p.name} ({pos_name}): CALL {actual_call:.1f}BB（全下）")
            else:
                print(f"\n{p.name} ({pos_name}): CALL {actual_call:.1f}BB")
            self.action_history.append({
                "player": "villain", "actor": player_idx, "action": "call",
                "amount": actual_call, "position": p.position, "street": self._current_street,
                "pot": pot,
            })
            return "call", actual_call, pot, last_raise_increment

        elif action in ("raise", "3bet", "4bet", "limp", "all-in", "allin"):
            if action in ("all-in", "allin"):
                raise_to = p.bet_street + p.stack
            else:
                # amount 是加注到的总额
                raise_to = max(amount, min_raise)

            additional = raise_to - p.bet_street
            additional = min(additional, p.stack)
            if additional <= 0:
                # 无法加注，改为跟注或过牌
                if to_call > 0:
                    actual_call = min(to_call, p.stack)
                    p.stack -= actual_call
                    p.bet_street += actual_call
                    pot += actual_call
                    print(f"\n{p.name} ({pos_name}): CALL {actual_call:.1f}BB")
                    self.action_history.append({
                        "player": "villain", "actor": player_idx, "action": "call",
                        "amount": actual_call, "position": p.position, "street": self._current_street,
                        "pot": pot,
                    })
                    return "call", actual_call, pot, last_raise_increment
                else:
                    print(f"\n{p.name} ({pos_name}): CHECK")
                    self.action_history.append({
                        "player": "villain", "actor": player_idx, "action": "check",
                        "amount": 0.0, "position": p.position, "street": self._current_street,
                        "pot": pot,
                    })
                    return "check", 0.0, pot, last_raise_increment

            p.stack -= additional
            old_bet = p.bet_street
            p.bet_street += additional
            pot += additional

            new_lri = p.bet_street - old_bet
            if new_lri <= 0:
                new_lri = last_raise_increment

            reasoning = decision.get("reasoning", "")
            if p.stack <= 0:
                p.is_allin = True
                print(f"\n{p.name} ({pos_name}): ALL-IN {p.bet_street:.1f}BB")
                if reasoning and self.show_thinking:
                    print(f'  理由: "{reasoning}"')
                self.action_history.append({
                    "player": "villain", "actor": player_idx, "action": "all-in",
                    "amount": p.bet_street, "position": p.position, "street": self._current_street,
                    "pot": pot,
                })
                return "allin", p.bet_street, pot, new_lri
            else:
                action_name = {"3bet": "3BET", "4bet": "4BET"}.get(action, "RAISE")
                print(f"\n{p.name} ({pos_name}): {action_name} {p.bet_street:.1f}BB")
                if reasoning and self.show_thinking:
                    print(f'  理由: "{reasoning}"')
                self.action_history.append({
                    "player": "villain", "actor": player_idx, "action": "raise",
                    "amount": p.bet_street, "position": p.position, "street": self._current_street,
                    "pot": pot,
                })
                return "raise", p.bet_street, pot, new_lri

        # 默认过牌
        print(f"\n{p.name} ({pos_name}): CHECK")
        self.action_history.append({
            "player": "villain", "actor": player_idx, "action": "check",
            "amount": 0.0, "position": p.position, "street": self._current_street,
            "pot": pot,
        })
        return "check", 0.0, pot, last_raise_increment

    # ── 摊牌 ──────────────────────────────────────────────────────────────

    def _showdown(self) -> HandResult6:
        """多人摊牌：计算 side pots 并分配。"""
        active_indices = self._active_players()

        print("\n=== 摊牌 ===")
        evals = {}
        for i in active_indices:
            p = self.players[i]
            all_cards = list(p.hole_cards) + self.board
            ev = evaluate_7(all_cards)
            desc = hand_rank_description(ev)
            evals[i] = ev
            tag = "你" if p.is_human else p.name
            print(f"{tag:<8} ({p.pos_name}): {cards_display(p.hole_cards)} → {desc}")

        # 计算 side pots
        side_pots = calculate_side_pots(self.players)
        pot_winners: List[Tuple[int, float, str]] = []

        for sp in side_pots:
            eligible = [i for i in sp.eligible_players if not self.players[i].folded]
            if not eligible:
                # 把这锅给最后一个活跃玩家（不应该发生）
                eligible = sp.eligible_players

            # 找牌最大的
            best_eval = max(evals[i] for i in eligible if i in evals)
            winners_in_pot = [i for i in eligible if i in evals and evals[i] == best_eval]

            share = sp.amount / len(winners_in_pot)
            for wi in winners_in_pot:
                desc = hand_rank_description(best_eval)
                pot_winners.append((wi, share, desc))

        # 显示结果
        print()
        for wi, share, desc in pot_winners:
            p = self.players[wi]
            tag = "你" if p.is_human else p.name
            if len(side_pots) > 1:
                print(f"🎉 {tag} 赢得 {share:.1f}BB! ({desc})")
            else:
                print(f"🎉 {tag} 赢得底池 {share:.1f}BB! ({desc})")

        return HandResult6(pot_winners=pot_winners, showdown=True)

    def _fold_result(self) -> HandResult6:
        """处理弃牌结果：唯一活跃玩家赢得底池。"""
        active = self._active_players()
        if len(active) == 1:
            wi = active[0]
            p = self.players[wi]
            tag = "你" if p.is_human else p.name
            print(f"\n  {tag} 赢得底池 {self.pot:.1f}BB（其余弃牌）。")
            return HandResult6(
                pot_winners=[(wi, self.pot, "弃牌获胜")],
                showdown=False,
                reason="其余玩家弃牌",
            )
        # 多人仍活跃但停止下注（不应该从这里调用）
        return self._showdown()

    # ── 主流程 ─────────────────────────────────────────────────────────────

    def play(self) -> HandResult6:
        """执行完整的一手牌，返回 HandResult6。"""
        self._show_hand_header()

        # 1. 下盲注
        sb_idx = next((i for i, p in enumerate(self.players) if p.position == Position.SB), None)
        bb_idx = next((i for i, p in enumerate(self.players) if p.position == Position.BB), None)

        if sb_idx is not None:
            sb_p = self.players[sb_idx]
            sb_paid = min(SB_AMOUNT, sb_p.stack)
            sb_p.stack -= sb_paid
            sb_p.bet_street = sb_paid
            self.pot += sb_paid

        if bb_idx is not None:
            bb_p = self.players[bb_idx]
            bb_paid = min(BB_AMOUNT, bb_p.stack)
            bb_p.stack -= bb_paid
            bb_p.bet_street = bb_paid
            self.pot += bb_paid

        # 显示人类玩家手牌
        human = self.players[self.human_idx]
        print(f"你的手牌: {cards_display(human.hole_cards)}")

        # 2. 翻前
        print(f"\n--- 翻前 ---")
        print(f"底池: {self.pot:.1f}BB")
        self._current_street = "preflop"
        continues = self._do_betting_round("preflop")

        if not continues:
            return self._fold_result()

        active = self._active_players()
        if len(active) <= 1:
            return self._fold_result()

        # 检查是否所有活跃玩家都 all-in
        def all_allin_or_one_can_act():
            can_act = [i for i in active if not self.players[i].is_allin]
            return len(can_act) <= 1

        # 3. 翻牌
        flop = self.deck.deal_n(3)
        self.board.extend(flop)
        self._current_street = "flop"
        self._show_board("flop")

        if not all_allin_or_one_can_act():
            continues = self._do_betting_round("flop")
            if not continues:
                return self._fold_result()

        active = self._active_players()
        if len(active) <= 1:
            return self._fold_result()

        # 4. 转牌
        turn = self.deck.deal()
        self.board.append(turn)
        self._current_street = "turn"
        self._show_board("turn")

        if not all_allin_or_one_can_act():
            continues = self._do_betting_round("turn")
            if not continues:
                return self._fold_result()

        active = self._active_players()
        if len(active) <= 1:
            return self._fold_result()

        # 5. 河牌
        river = self.deck.deal()
        self.board.append(river)
        self._current_street = "river"
        self._show_board("river")

        if not all_allin_or_one_can_act():
            continues = self._do_betting_round("river")
            if not continues:
                return self._fold_result()

        active = self._active_players()
        if len(active) <= 1:
            return self._fold_result()

        # 6. 摊牌
        return self._showdown()


# ---------------------------------------------------------------------------
# 主游戏循环
# ---------------------------------------------------------------------------

def build_players(btn_seat_idx: int, stacks: Dict[str, float]) -> List[PlayerState6]:
    """
    根据 BTN 的座位号，构建 6 个玩家的 PlayerState6 列表。
    btn_seat_idx: 0-5，表示 positions 列表中哪个是 BTN
    返回按座位顺序的玩家列表（座位0=UTG, ..., 座位5=BB）。
    HUMAN_SEAT=3 始终是人类玩家，其余是 Bot-1..5
    """
    # 人类玩家名称
    human_name = "你"
    # 6个座位的名称（固定）
    seat_names = ["Bot-1", "Bot-2", "Bot-3", human_name, "Bot-4", "Bot-5"]

    # 确定每个座位的位置（轮转）
    # positions 列表按桌子顺序，btn_seat_idx 表示 BTN 坐在哪个座位
    # 每个座位的位置 = (SIX_MAX_POSITIONS 中 BTN 的偏移)
    # BTN=Position.BTN(3), 座位0相对于BTN的偏移
    # 如果 btn_seat=3，则 seat0=UTG, seat1=HJ, seat2=CO, seat3=BTN, seat4=SB, seat5=BB

    # BTN 位置在 SIX_MAX_POSITIONS 中的索引是 3（Position.BTN）
    btn_pos_idx = 3  # BTN 在 SIX_MAX_POSITIONS 中的位置

    players = []
    for seat in range(6):
        # 相对偏移
        offset = (seat - btn_seat_idx) % 6
        pos = SIX_MAX_POSITIONS[(btn_pos_idx + offset) % 6]
        name = seat_names[seat]
        is_human = (name == human_name)
        stack = stacks.get(name, STARTING_STACK)
        players.append(PlayerState6(
            name=name,
            position=pos,
            stack=stack,
            is_human=is_human,
        ))

    return players


def main() -> None:
    banner()

    # 初始筹码
    stacks: Dict[str, float] = {
        "Bot-1": STARTING_STACK,
        "Bot-2": STARTING_STACK,
        "Bot-3": STARTING_STACK,
        "你":    STARTING_STACK,
        "Bot-4": STARTING_STACK,
        "Bot-5": STARTING_STACK,
    }

    # 活跃玩家集合（被淘汰的移出）
    active_names = list(stacks.keys())

    # 显示开始界面
    print(f"起始筹码: {STARTING_STACK:.0f}BB")
    print(f"盲注: {SB_AMOUNT}BB / {BB_AMOUNT}BB")
    print("输入 'q' 随时退出 | 'h' 查看帮助")
    print()

    # 初始座位表（BTN 从随机位置开始）
    btn_seat = random.randint(0, 5)

    # 构建初始玩家列表（仅用于显示开始界面）
    players_preview = build_players(btn_seat, stacks)
    print("座位表:")
    for p in players_preview:
        sb_bb = ""
        if p.position == Position.SB:
            sb_bb = f" (SB)"
        elif p.position == Position.BB:
            sb_bb = f" (BB)"
        marker = "  ← 你在这里" if p.is_human else ""
        print(f"  Seat {players_preview.index(p)+1} ({p.pos_name}): {p.name:<8} [{p.stack:.1f}BB]{sb_bb}{marker}")
    print()

    try:
        input("按 Enter 开始...")
    except (EOFError, KeyboardInterrupt):
        print("\n退出游戏。")
        return

    # 创建 5 个 Bot 实例（每个 Bot 固定坐某个位置，但位置会轮转）
    bot_names = [n for n in active_names if n != "你"]
    bots: Dict[str, PokerBot] = {}
    for bn in bot_names:
        # 初始位置根据座位决定，后续每手更新
        bots[bn] = PokerBot(hero_position=Position.UTG)

    show_thinking = True
    hand_num = 0
    wins: Dict[str, int] = {n: 0 for n in active_names}

    while True:
        # 检查游戏结束条件
        active_names = [n for n in stacks if stacks[n] >= BB_AMOUNT]
        human_active = "你" in active_names

        if not human_active:
            print("\n╔═══════════════════════════════╗")
            print("║     💀 游戏结束!              ║")
            print("║     你筹码耗尽，游戏结束！     ║")
            print(f"║     总共打了 {hand_num:<18}║")
            print("║     继续练习！                 ║")
            print("╚═══════════════════════════════╝")
            break

        if len(active_names) < 2:
            print("\n╔═══════════════════════════════╗")
            print("║     🏆 游戏结束!              ║")
            print("║     你赢了！所有 Bot 筹码归零  ║")
            print(f"║     总共打了 {hand_num:<18}║")
            print("╚═══════════════════════════════╝")
            break

        # 构建本手玩家列表（只有活跃玩家参与）
        # 重新映射座位：活跃玩家按原座位顺序，BTN 轮转
        current_seat_names = ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]
        current_active_seats = [n for n in current_seat_names if n in active_names]
        n_active = len(current_active_seats)

        if n_active < 2:
            break

        # BTN 座位在活跃玩家中轮转
        # btn_seat 是在原始 6 个座位中的索引
        # 需要映射到活跃玩家的索引
        btn_seat = btn_seat % 6

        hand_num += 1

        # 构建玩家列表（使用当前 stacks）
        if n_active == 6:
            players = build_players(btn_seat, stacks)
        else:
            # 少于 6 人时简化处理：重新映射位置
            players = _build_players_reduced(current_active_seats, btn_seat, stacks)

        # 更新每个 bot 的当前位置
        for p in players:
            if p.name in bots:
                bots[p.name].hero_position = p.position
                bots[p.name].reset_hand()

        # 执行一手牌
        engine = HandEngine6Max(
            players=players,
            bots=bots,
            show_thinking=show_thinking,
            hand_num=hand_num,
        )

        result = engine.play()

        # 更新筹码
        # 先将 engine 中玩家的剩余筹码更新到 stacks
        for p in engine.players:
            stacks[p.name] = p.stack

        # 分配底池
        for (wi, share, desc) in result.pot_winners:
            wp = engine.players[wi]
            stacks[wp.name] += share
            wins[wp.name] = wins.get(wp.name, 0) + 1
            if wp.is_human:
                if len(result.pot_winners) == 1:
                    print(f"\n🎉 你赢了 {share:.1f}BB!")
            else:
                if not result.showdown and len(result.pot_winners) == 1:
                    print(f"\n😞 {wp.name} 赢了 {share:.1f}BB!")

        # 检查淘汰
        for name in list(active_names):
            if name != "你" and stacks.get(name, 0) < BB_AMOUNT:
                print(f"\n💀 {name} 筹码耗尽，离开牌桌！")
                del stacks[name]
                wins.pop(name, None)
                if name in bots:
                    del bots[name]

        # 更新活跃名单
        remaining_active = [n for n in stacks if stacks[n] >= BB_AMOUNT]
        if len(remaining_active) == 2 and "你" in remaining_active:
            print(f"\n⚠️  只剩 2 人，进入 Heads-Up 模式！")

        # 显示筹码
        print(f"\n当前筹码:")
        for name in ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]:
            if name in stacks:
                tag = "你" if name == "你" else name
                print(f"  {tag}: {stacks[name]:.1f}BB")

        print("━" * 40)

        # 每 10 手显示统计
        if hand_num % 10 == 0:
            # 构建临时玩家列表用于统计显示
            stat_players = [
                PlayerState6(name=n, position=Position.UTG, stack=stacks.get(n, 0),
                             is_human=(n == "你"))
                for n in ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]
                if n in stacks
            ]
            show_stats_6max(hand_num, stat_players, wins, STARTING_STACK, "你")

        # 进入下一手前询问
        try:
            raw = input("\n按 Enter 继续下一手... (或输入 'q' 退出, 's' 统计, 'thinking' 切换思考) ")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = raw.strip().lower()
        if cmd in ("q", "quit"):
            break
        elif cmd in ("s", "status"):
            stat_players = [
                PlayerState6(name=n, position=Position.UTG, stack=stacks.get(n, 0),
                             is_human=(n == "你"))
                for n in ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]
                if n in stacks
            ]
            show_stats_6max(hand_num, stat_players, wins, STARTING_STACK, "你")
        elif cmd == "thinking":
            show_thinking = not show_thinking
            state = "开启" if show_thinking else "关闭"
            print(f"  Bot 思考过程已{state}。")

        # BTN 轮转：在原始6座位中顺时针转
        btn_seat = (btn_seat + 1) % 6

    # 退出时显示最终统计
    stat_players = [
        PlayerState6(name=n, position=Position.UTG, stack=stacks.get(n, 0),
                     is_human=(n == "你"))
        for n in ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]
        if n in stacks
    ]
    show_stats_6max(hand_num, stat_players, wins, STARTING_STACK, "你")
    print("感谢游玩！再见！")


def _build_players_reduced(
    active_seats: List[str],
    btn_seat: int,
    stacks: Dict[str, float],
) -> List[PlayerState6]:
    """
    构建少于 6 人时的玩家列表。
    active_seats: 活跃玩家的名称列表（按原座位顺序）
    btn_seat: BTN 在原始 6 座位中的索引
    """
    # 原始座位顺序
    all_seat_names = ["Bot-1", "Bot-2", "Bot-3", "你", "Bot-4", "Bot-5"]
    n = len(active_seats)

    # 找到 BTN 在活跃玩家中的位置
    # btn_seat 在原始座位中，找最近的活跃玩家作为 BTN
    btn_name = all_seat_names[btn_seat % 6]
    if btn_name not in active_seats:
        # 找顺时针最近的活跃玩家作为 BTN
        for offset in range(1, 6):
            candidate = all_seat_names[(btn_seat + offset) % 6]
            if candidate in active_seats:
                btn_name = candidate
                break

    btn_active_idx = active_seats.index(btn_name)

    # 根据人数确定位置
    if n == 2:
        # Heads-up: BTN=SB, BB
        positions_for_n = [Position.BTN, Position.BB]
    elif n == 3:
        positions_for_n = [Position.BTN, Position.SB, Position.BB]
    elif n == 4:
        positions_for_n = [Position.CO, Position.BTN, Position.SB, Position.BB]
    elif n == 5:
        positions_for_n = [Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]
    else:
        positions_for_n = [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB]

    players = []
    for i, name in enumerate(active_seats):
        # 相对于 BTN 的偏移
        offset = (i - btn_active_idx) % n
        pos = positions_for_n[offset % len(positions_for_n)]
        players.append(PlayerState6(
            name=name,
            position=pos,
            stack=stacks.get(name, STARTING_STACK),
            is_human=(name == "你"),
        ))

    return players


if __name__ == "__main__":
    main()
