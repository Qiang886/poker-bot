#!/usr/bin/env python3
"""
play.py — 6-Max NLHE 命令行对战模式
运行方式: python play.py
"""

import sys
import os
import random

# 确保能找到 src/ 目录
sys.path.insert(0, os.path.dirname(__file__))

from src.card import Card, Rank, Suit
from src.position import Position
from src.bot import PokerBot, GameState
from src.evaluator import evaluate_7
from src.hand_analysis import classify_hand
from src.game_engine import (
    GameDeck, PlayerState, HandResult, ActionRecord,
    card_display, cards_display, hand_rank_description,
    clamp_raise, calc_min_raise, STREETS,
)

# ---------------------------------------------------------------------------
# 全局游戏配置
# ---------------------------------------------------------------------------
STARTING_STACK = 100.0   # BB
SB_AMOUNT = 0.5
BB_AMOUNT = 1.0


# ---------------------------------------------------------------------------
# 显示工具
# ---------------------------------------------------------------------------

def clear_line() -> None:
    print()


def banner() -> None:
    print("╔══════════════════════════════════════╗")
    print("║     6-Max NLHE Poker Bot v2          ║")
    print("║     命令行对战模式                    ║")
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
    print("  a / all / allin   全下")
    print("  s / status        显示当前状态")
    print("  thinking          切换显示 Bot 思考过程")
    print("  h / help          显示帮助")
    print("  q / quit          退出游戏")
    print("────────────────────────────────────────\n")


def show_stats(hand_num: int, player_wins: int, bot_wins: int,
               player_stack: float, bot_stack: float,
               starting_stack: float) -> None:
    total_hands = player_wins + bot_wins
    bb100 = 0
    if total_hands > 0:
        profit = player_stack - starting_stack
        bb100 = int(profit / total_hands * 100)
    print()
    print("╔═══════════════════════════════╗")
    print("║        对战统计               ║")
    print("╠═══════════════════════════════╣")
    print(f"║  总手数: {hand_num:<21}║")
    print(f"║  你赢: {player_wins} 手 | Bot 赢: {bot_wins} 手{' ' * max(0, 9 - len(str(player_wins)) - len(str(bot_wins)))}║")
    print(f"║  你的筹码: {player_stack:.1f}BB{' ' * max(0, 17 - len(f'{player_stack:.1f}'))}║")
    print(f"║  Bot 筹码: {bot_stack:.1f}BB{' ' * max(0, 17 - len(f'{bot_stack:.1f}'))}║")
    print(f"║  你的 bb/100: {bb100:+d}{' ' * max(0, 14 - len(str(abs(bb100))))}║")
    print("╚═══════════════════════════════╝")
    print()


def show_game_over(winner: str, hand_num: int) -> None:
    if winner == "你":
        print("╔═══════════════════════════════╗")
        print("║     🏆 游戏结束!              ║")
        print("║     你赢了！Bot 筹码归零       ║")
        print(f"║     总共打了 {hand_num} 手{' ' * max(0, 15 - len(str(hand_num)))}║")
        print("╚═══════════════════════════════╝")
    else:
        print("╔═══════════════════════════════╗")
        print("║     💀 游戏结束!              ║")
        print("║     Bot 赢了... 你筹码归零     ║")
        print(f"║     总共打了 {hand_num} 手{' ' * max(0, 15 - len(str(hand_num)))}║")
        print("║     继续练习！                 ║")
        print("╚═══════════════════════════════╝")


# ---------------------------------------------------------------------------
# 用户输入解析
# ---------------------------------------------------------------------------

def parse_action(raw: str, to_call: float, player_stack: float,
                 pot: float, min_raise: float,
                 show_thinking_ref: list) -> tuple:
    """
    解析用户输入，返回 (action, amount) 或 (None, None) 表示无效输入。
    特殊返回:
      ("quit", 0)
      ("help", 0)
      ("status", 0)
      ("toggle_thinking", 0)
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

    # raise / r / bet / b <amount>
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
# Bot 决策辅助
# ---------------------------------------------------------------------------

def build_bot_game_state(
    bot_hand: tuple,
    board: list,
    pot: float,
    to_call: float,
    bot_stack: float,
    player_stack: float,
    bot_position: Position,
    player_position: Position,
    street: str,
    action_history: list,
) -> GameState:
    return GameState(
        hero_hand=bot_hand,
        board=board,
        pot=pot,
        to_call=to_call,
        hero_stack=bot_stack,
        villain_stacks=[player_stack],
        hero_position=bot_position,
        villain_positions=[player_position],
        street=street,
        action_history=action_history,
        is_tournament=False,
        num_players=2,
    )


def format_bot_thinking(decision: dict, bot_position: Position,
                        bot_hand: tuple, board: list) -> str:
    """格式化 bot 思考过程显示。"""
    lines = ["Bot 思考中..."]
    pos_name = bot_position.name
    lines.append(f"  位置: {pos_name}")

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
    elif action == "allin" or action == "all-in":
        lines.append(f"  决定: ALL-IN")
    else:
        lines.append(f"  决定: {action.upper()}")

    reasoning = decision.get("reasoning", "")
    if reasoning:
        lines.append(f"  理由: \"{reasoning}\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 单手牌游戏逻辑
# ---------------------------------------------------------------------------

class HandEngine:
    """
    管理一手牌的完整流程：发牌 → 翻前 → 翻牌 → 转牌 → 河牌 → 结算。
    """

    def __init__(
        self,
        player_stack: float,
        bot_stack: float,
        player_is_btn: bool,     # True = 玩家在 BTN/SB，False = 玩家在 BB
        bot: PokerBot,
        show_thinking: bool,
        hand_num: int,
    ) -> None:
        self.player_stack = player_stack
        self.bot_stack = bot_stack
        self.player_is_btn = player_is_btn
        self.bot = bot
        self.show_thinking = show_thinking
        self.hand_num = hand_num

        # Heads-up: BTN = SB，BB 翻前后行动
        if player_is_btn:
            self.player_position = Position.BTN
            self.bot_position = Position.BB
        else:
            self.player_position = Position.BB
            self.bot_position = Position.BTN

        # 发牌
        deck = GameDeck()
        # 先给玩家发 2 张，再给 bot 发 2 张
        self.player_hand: tuple = (deck.deal(), deck.deal())
        self.bot_hand: tuple = (deck.deal(), deck.deal())
        self.deck = deck

        self.board: list = []
        self.pot: float = 0.0
        # 每个玩家本街已投入的筹码
        self.player_street_bet: float = 0.0
        self.bot_street_bet: float = 0.0

        # 行动历史（供 bot 决策使用）
        self.action_history: list = []

        # 标记弃牌
        self.player_folded: bool = False
        self.bot_folded: bool = False

        # 追踪上一次加注增量（用于最小加注计算）
        self.last_raise_increment: float = BB_AMOUNT

    # ── 辅助 ──────────────────────────────────────────────────────────────

    def _deduct(self, who: str, amount: float) -> float:
        """从筹码中扣除金额，返回实际扣除量（可能因筹码不足而减少）。"""
        if who == "player":
            actual = min(amount, self.player_stack)
            self.player_stack -= actual
            return actual
        else:
            actual = min(amount, self.bot_stack)
            self.bot_stack -= actual
            return actual

    def _add_to_pot(self, amount: float) -> None:
        self.pot += amount

    def _reset_street_bets(self) -> None:
        self.player_street_bet = 0.0
        self.bot_street_bet = 0.0
        self.last_raise_increment = BB_AMOUNT

    def _player_to_call(self) -> float:
        """玩家需要跟注的金额（本街差额）。"""
        return max(0.0, self.bot_street_bet - self.player_street_bet)

    def _bot_to_call(self) -> float:
        """Bot 需要跟注的金额（本街差额）。"""
        return max(0.0, self.player_street_bet - self.bot_street_bet)

    def _show_board(self, street: str) -> None:
        street_names = {
            "flop": "翻牌",
            "turn": "转牌",
            "river": "河牌",
        }
        print(f"\n--- {street_names.get(street, street)} ---")
        print(f"牌面: {cards_display(self.board)}")
        print(f"底池: {self.pot:.1f}BB")

    # ── 下注街通用逻辑 ────────────────────────────────────────────────────

    def _betting_round(self, street: str, player_acts_first: bool) -> bool:
        """
        执行一整轮下注。
        返回 True 继续游戏，False 表示有人弃牌或游戏结束。
        """
        # 翻前盲注已在 play() 里 post，不重置；翻后各街重置
        if street != "preflop":
            self._reset_street_bets()

        first_actor = "player" if player_acts_first else "bot"
        second_actor = "bot" if player_acts_first else "player"

        # 双方都行动过的标记
        acted = {"player": False, "bot": False}

        current_actor = first_actor

        while True:
            # 检查是否两人都 check 或跟注完毕
            if acted[first_actor] and acted[second_actor]:
                # 检查下注额是否相等
                if abs(self.player_street_bet - self.bot_street_bet) < 0.001:
                    break

            # 如果当前行动者已 allin 或弃牌，切换
            if current_actor == "player" and (self.player_folded or self.player_stack <= 0):
                break
            if current_actor == "bot" and (self.bot_folded or self.bot_stack <= 0):
                break

            if current_actor == "player":
                result = self._player_action(street)
                if result == "fold":
                    self.player_folded = True
                    return False
                acted["player"] = True
                if self.player_stack <= 0:
                    # player allin
                    current_actor = second_actor
                    if not acted[second_actor]:
                        continue
                    break
            else:
                result = self._bot_action(street)
                if result == "fold":
                    self.bot_folded = True
                    return False
                acted["bot"] = True
                if self.bot_stack <= 0:
                    # bot allin
                    current_actor = first_actor
                    if not acted[first_actor]:
                        continue
                    break

            # 如果有人加注，对方需要重新行动
            if result == "raise":
                # 切换到对方
                current_actor = second_actor
                acted[second_actor] = False  # 对方需要重新行动
                second_actor, first_actor = first_actor, second_actor
            else:
                # 切换到对方继续
                if not acted[second_actor]:
                    current_actor = second_actor
                    second_actor, first_actor = first_actor, second_actor
                else:
                    break

        return True

    # ── 玩家行动 ──────────────────────────────────────────────────────────

    def _player_action(self, street: str) -> str:
        """提示玩家输入行动，返回 "fold"/"call"/"check"/"raise"/"allin"。"""
        to_call = self._player_to_call()
        min_raise = calc_min_raise(
            self.bot_street_bet, self.last_raise_increment, to_call
        )
        max_raise = self.player_street_bet + to_call + self.player_stack

        # 构建提示
        options = []
        if to_call > 0:
            options.append(f"[f]old")
            actual_call = min(to_call, self.player_stack)
            options.append(f"[c]all {actual_call:.1f}BB")
        else:
            options.append(f"[ch]eck")

        if self.player_stack > to_call:
            if to_call > 0:
                options.append(f"[r]aise 到多少BB (最小 {min_raise:.1f}BB)")
            else:
                options.append(f"[b]et 多少BB")
        options.append(f"[a]llin ({self.player_stack + self.player_street_bet:.1f}BB)")

        prompt = " / ".join(options)

        while True:
            try:
                raw = input(f"\n> 你的选择 {prompt}? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出游戏。")
                sys.exit(0)

            action, amount = parse_action(
                raw, to_call, self.player_stack, self.pot, min_raise, []
            )

            if action == "quit":
                print("\n退出游戏。")
                sys.exit(0)
            elif action == "help":
                show_help()
                continue
            elif action == "status":
                print(f"\n  底池: {self.pot:.1f}BB | 你: {self.player_stack:.1f}BB | Bot: {self.bot_stack:.1f}BB")
                print(f"  牌面: {cards_display(self.board) if self.board else '(翻前)'}")
                print(f"  你的手牌: {cards_display(list(self.player_hand))}")
                continue
            elif action == "toggle_thinking":
                self.show_thinking = not self.show_thinking
                state = "开启" if self.show_thinking else "关闭"
                print(f"  Bot 思考过程已{state}。")
                continue
            elif action is None:
                continue

            # 执行合法行动
            if action == "fold":
                print(f"  你选择弃牌。")
                self.action_history.append({
                    "player": "hero", "action": "fold",
                    "amount": 0.0, "position": self.player_position, "street": street
                })
                return "fold"

            elif action == "check":
                print(f"  你选择过牌。")
                self.action_history.append({
                    "player": "hero", "action": "check",
                    "amount": 0.0, "position": self.player_position, "street": street
                })
                return "check"

            elif action == "call":
                actual_call = min(to_call, self.player_stack)
                paid = self._deduct("player", actual_call)
                self.player_street_bet += paid
                self._add_to_pot(paid)
                if self.player_stack <= 0:
                    print(f"  你跟注 {paid:.1f}BB（全下）。")
                else:
                    print(f"  你跟注 {paid:.1f}BB。")
                self.action_history.append({
                    "player": "hero", "action": "call",
                    "amount": paid, "position": self.player_position, "street": street
                })
                return "call"

            elif action in ("raise", "allin"):
                if action == "allin":
                    raise_to = self.player_street_bet + self.player_stack
                else:
                    raise_to = amount

                # 实际能投入的
                additional = raise_to - self.player_street_bet
                additional = min(additional, self.player_stack)
                paid = self._deduct("player", additional)
                old_street_bet = self.player_street_bet
                self.player_street_bet += paid
                self._add_to_pot(paid)

                # 更新最小加注增量
                new_increment = self.player_street_bet - self.bot_street_bet
                if new_increment > 0:
                    self.last_raise_increment = new_increment

                if self.player_stack <= 0:
                    print(f"  你全下: {self.player_street_bet:.1f}BB。")
                    self.action_history.append({
                        "player": "hero", "action": "all-in",
                        "amount": self.player_street_bet, "position": self.player_position, "street": street
                    })
                    return "raise"
                else:
                    print(f"  你加注到 {self.player_street_bet:.1f}BB。")
                    self.action_history.append({
                        "player": "hero", "action": "raise",
                        "amount": self.player_street_bet, "position": self.player_position, "street": street
                    })
                    return "raise"

        return "check"  # 不可达，但让 linter 满意

    # ── Bot 行动 ──────────────────────────────────────────────────────────

    def _bot_action(self, street: str) -> str:
        """Bot 决策并执行，返回 "fold"/"call"/"check"/"raise"/"allin"。"""
        to_call = self._bot_to_call()

        # 构建 GameState
        gs = build_bot_game_state(
            bot_hand=self.bot_hand,
            board=self.board,
            pot=self.pot,
            to_call=to_call,
            bot_stack=self.bot_stack,
            player_stack=self.player_stack,
            bot_position=self.bot_position,
            player_position=self.player_position,
            street=street,
            action_history=self.action_history,
        )

        decision = self.bot.decide(gs)
        action = decision.get("action", "fold")
        amount = decision.get("amount", 0.0)

        if self.show_thinking:
            print(format_bot_thinking(decision, self.bot_position, self.bot_hand, self.board))

        # 执行 bot 行动
        if action == "fold":
            reasoning = decision.get("reasoning", "")
            print(f"\nBot: FOLD")
            if reasoning:
                print(f'  理由: "{reasoning}"')
            self.action_history.append({
                "player": "villain", "action": "fold",
                "amount": 0.0, "position": self.bot_position, "street": street
            })
            return "fold"

        elif action in ("check",):
            print(f"\nBot: CHECK")
            self.action_history.append({
                "player": "villain", "action": "check",
                "amount": 0.0, "position": self.bot_position, "street": street
            })
            return "check"

        elif action == "call":
            actual_call = min(to_call, self.bot_stack)
            paid = self._deduct("bot", actual_call)
            self.bot_street_bet += paid
            self._add_to_pot(paid)
            if self.bot_stack <= 0:
                print(f"\nBot: CALL {paid:.1f}BB（全下）")
            else:
                print(f"\nBot: CALL {paid:.1f}BB")
            self.action_history.append({
                "player": "villain", "action": "call",
                "amount": paid, "position": self.bot_position, "street": street
            })
            return "call"

        elif action in ("raise", "3bet", "4bet", "limp"):
            # amount 是加注到的总金额（BB 为单位）
            # 确保不低于最小加注
            min_raise = calc_min_raise(
                self.player_street_bet, self.last_raise_increment, to_call
            )
            raise_to_total = max(amount, min_raise)
            additional = raise_to_total - self.bot_street_bet
            additional = min(additional, self.bot_stack)
            if additional <= 0:
                # 无法加注，改为 call 或 check
                if to_call > 0:
                    actual_call = min(to_call, self.bot_stack)
                    paid = self._deduct("bot", actual_call)
                    self.bot_street_bet += paid
                    self._add_to_pot(paid)
                    print(f"\nBot: CALL {paid:.1f}BB")
                    self.action_history.append({
                        "player": "villain", "action": "call",
                        "amount": paid, "position": self.bot_position, "street": street
                    })
                    return "call"
                else:
                    print(f"\nBot: CHECK")
                    self.action_history.append({
                        "player": "villain", "action": "check",
                        "amount": 0.0, "position": self.bot_position, "street": street
                    })
                    return "check"

            paid = self._deduct("bot", additional)
            old_bot_bet = self.bot_street_bet
            self.bot_street_bet += paid
            self._add_to_pot(paid)

            new_increment = self.bot_street_bet - self.player_street_bet
            if new_increment > 0:
                self.last_raise_increment = new_increment

            reasoning = decision.get("reasoning", "")
            if self.bot_stack <= 0:
                print(f"\nBot: ALL-IN {self.bot_street_bet:.1f}BB")
                if reasoning and self.show_thinking:
                    print(f'  理由: "{reasoning}"')
                self.action_history.append({
                    "player": "villain", "action": "all-in",
                    "amount": self.bot_street_bet, "position": self.bot_position, "street": street
                })
            else:
                action_name = {"3bet": "3BET", "4bet": "4BET"}.get(action, "RAISE")
                print(f"\nBot: {action_name} {self.bot_street_bet:.1f}BB")
                if reasoning and self.show_thinking:
                    print(f'  理由: "{reasoning}"')
                self.action_history.append({
                    "player": "villain", "action": "raise",
                    "amount": self.bot_street_bet, "position": self.bot_position, "street": street
                })
            return "raise"

        elif action in ("all-in", "allin"):
            additional = self.bot_stack
            paid = self._deduct("bot", additional)
            self.bot_street_bet += paid
            self._add_to_pot(paid)

            new_increment = self.bot_street_bet - self.player_street_bet
            if new_increment > 0:
                self.last_raise_increment = new_increment

            reasoning = decision.get("reasoning", "")
            print(f"\nBot: ALL-IN {self.bot_street_bet:.1f}BB")
            if reasoning and self.show_thinking:
                print(f'  理由: "{reasoning}"')
            self.action_history.append({
                "player": "villain", "action": "all-in",
                "amount": self.bot_street_bet, "position": self.bot_position, "street": street
            })
            return "raise"

        # 默认 check
        print(f"\nBot: CHECK")
        self.action_history.append({
            "player": "villain", "action": "check",
            "amount": 0.0, "position": self.bot_position, "street": street
        })
        return "check"

    # ── 摊牌 ──────────────────────────────────────────────────────────────

    def _showdown(self) -> HandResult:
        """摊牌：比较双方手牌，决定赢家。"""
        all_cards_player = list(self.player_hand) + self.board
        all_cards_bot = list(self.bot_hand) + self.board

        player_eval = evaluate_7(all_cards_player)
        bot_eval = evaluate_7(all_cards_bot)

        player_desc = hand_rank_description(player_eval)
        bot_desc = hand_rank_description(bot_eval)

        print("\n=== 摊牌 ===")
        print(f"你的手牌: {cards_display(list(self.player_hand))} → {player_desc}")
        print(f"Bot 手牌: {cards_display(list(self.bot_hand))} → {bot_desc}")

        if player_eval > bot_eval:
            winner = "你"
            amount = self.pot
        elif bot_eval > player_eval:
            winner = "Bot"
            amount = self.pot
        else:
            winner = "平局"
            amount = self.pot / 2

        return HandResult(
            winner=winner,
            amount=amount,
            showdown=True,
            player_best=player_desc,
            bot_best=bot_desc,
        )

    # ── 主流程 ─────────────────────────────────────────────────────────────

    def play(self) -> HandResult:
        """执行完整的一手牌，返回 HandResult。"""
        print(f"\n{'━' * 10} 第 {self.hand_num} 手 {'━' * 10}")

        player_pos_name = self.player_position.name
        bot_pos_name = self.bot_position.name
        print(f"你的位置: {player_pos_name} | Bot 位置: {bot_pos_name}")
        print(f"你的筹码: {self.player_stack:.1f}BB | Bot 筹码: {self.bot_stack:.1f}BB")

        # ── 1. 下盲注 ──────────────────────────────────────────────────
        # Heads-up: BTN = SB，先行动 (preflop)
        if self.player_is_btn:
            # 玩家 SB
            sb_paid = self._deduct("player", SB_AMOUNT)
            self.player_street_bet = sb_paid
            self._add_to_pot(sb_paid)
            bb_paid = self._deduct("bot", BB_AMOUNT)
            self.bot_street_bet = bb_paid
            self._add_to_pot(bb_paid)
            print(f"盲注已下: SB {sb_paid:.1f}BB / BB {bb_paid:.1f}BB")
        else:
            # Bot SB
            sb_paid = self._deduct("bot", SB_AMOUNT)
            self.bot_street_bet = sb_paid
            self._add_to_pot(sb_paid)
            bb_paid = self._deduct("player", BB_AMOUNT)
            self.player_street_bet = bb_paid
            self._add_to_pot(bb_paid)
            print(f"盲注已下: SB {sb_paid:.1f}BB / BB {bb_paid:.1f}BB")

        # 显示手牌
        print(f"\n你的手牌: {cards_display(list(self.player_hand))}")

        # ── 2. 翻前 ────────────────────────────────────────────────────
        print(f"\n--- 翻前 ---")
        print(f"底池: {self.pot:.1f}BB")

        # 翻前：BTN/SB 先行动
        player_first_preflop = self.player_is_btn
        continued = self._betting_round("preflop", player_acts_first=player_first_preflop)
        if not continued:
            return self._fold_result()

        # 如果有人全下且另一方跟注，直接到摊牌
        both_allin = (self.player_stack <= 0 and self.bot_stack <= 0)

        # ── 3. 发翻牌 ──────────────────────────────────────────────────
        if self.player_folded or self.bot_folded:
            return self._fold_result()

        flop_cards = self.deck.deal_n(3)
        self.board.extend(flop_cards)
        self._show_board("flop")

        if not both_allin:
            # 翻后：BB 先行动
            player_first_postflop = not self.player_is_btn
            continued = self._betting_round("flop", player_acts_first=player_first_postflop)
            if not continued:
                return self._fold_result()

        both_allin = (self.player_stack <= 0 and self.bot_stack <= 0)

        # ── 4. 发转牌 ──────────────────────────────────────────────────
        if self.player_folded or self.bot_folded:
            return self._fold_result()

        turn_card = self.deck.deal()
        self.board.append(turn_card)
        self._show_board("turn")

        if not both_allin:
            player_first_postflop = not self.player_is_btn
            continued = self._betting_round("turn", player_acts_first=player_first_postflop)
            if not continued:
                return self._fold_result()

        both_allin = (self.player_stack <= 0 and self.bot_stack <= 0)

        # ── 5. 发河牌 ──────────────────────────────────────────────────
        if self.player_folded or self.bot_folded:
            return self._fold_result()

        river_card = self.deck.deal()
        self.board.append(river_card)
        self._show_board("river")

        if not both_allin:
            player_first_postflop = not self.player_is_btn
            continued = self._betting_round("river", player_acts_first=player_first_postflop)
            if not continued:
                return self._fold_result()

        # ── 6. 摊牌 ────────────────────────────────────────────────────
        if self.player_folded or self.bot_folded:
            return self._fold_result()

        return self._showdown()

    def _fold_result(self) -> HandResult:
        """处理弃牌结果。"""
        if self.player_folded:
            return HandResult(
                winner="Bot",
                amount=self.pot,
                showdown=False,
                reason="你弃牌",
            )
        else:
            reasoning = ""
            # 找最后一条 bot fold 记录的理由
            for rec in reversed(self.action_history):
                if rec.get("player") == "villain" and rec.get("action") == "fold":
                    break
            reasoning = ""
            # 从 bot decision reasoning 中取（通过 action_history 无法直接获取，这里留空）
            return HandResult(
                winner="你",
                amount=self.pot,
                showdown=False,
                reason="Bot 弃牌",
            )


# ---------------------------------------------------------------------------
# 主游戏循环
# ---------------------------------------------------------------------------

def main() -> None:
    banner()
    print(f"起始筹码: {STARTING_STACK:.0f}BB")
    print(f"盲注: {SB_AMOUNT}BB / {BB_AMOUNT}BB")
    print("输入 'q' 随时退出")
    print("输入 'h' 查看帮助")
    print()

    try:
        input("按 Enter 开始...")
    except (EOFError, KeyboardInterrupt):
        print("\n退出游戏。")
        return

    player_stack = STARTING_STACK
    bot_stack = STARTING_STACK
    show_thinking = True    # 默认显示 bot 思考
    hand_num = 0
    player_wins = 0
    bot_wins = 0

    # Bot 实例（heads-up 用 BTN 作为默认位置，每手牌动态传入实际位置）
    bot = PokerBot(hero_position=Position.BB)

    # 随机决定第一手谁在 BTN
    player_is_btn = random.choice([True, False])

    while True:
        # 检查游戏结束条件
        if player_stack < BB_AMOUNT:
            show_game_over("Bot", hand_num)
            break
        if bot_stack < BB_AMOUNT:
            show_game_over("你", hand_num)
            break

        hand_num += 1

        # 重置 bot 的每手状态
        bot.reset_hand()
        bot.hero_position = Position.BB if player_is_btn else Position.BTN

        # 执行一手牌
        engine = HandEngine(
            player_stack=player_stack,
            bot_stack=bot_stack,
            player_is_btn=player_is_btn,
            bot=bot,
            show_thinking=show_thinking,
            hand_num=hand_num,
        )

        result = engine.play()

        # 更新筹码（engine 里已经扣了盲注，pot 是总底池）
        # 先把各自已扣除的筹码加回来，再重新分配底池
        # 注意：engine.player_stack 和 engine.bot_stack 已经是扣完后的余额
        player_stack = engine.player_stack
        bot_stack = engine.bot_stack

        # 分配底池
        if result.winner == "你":
            player_stack += result.amount
            player_wins += 1
            print(f"\n🎉 你赢了 {result.amount:.1f}BB!")
        elif result.winner == "Bot":
            bot_stack += result.amount
            bot_wins += 1
            print(f"\n😞 Bot 赢了 {result.amount:.1f}BB!")
        else:
            # 平局
            player_stack += result.amount
            bot_stack += result.amount
            print(f"\n🤝 平局！各返还 {result.amount:.1f}BB。")

        if result.reason:
            print(f"  {result.reason}")

        print(f"\n当前筹码 → 你: {player_stack:.1f}BB | Bot: {bot_stack:.1f}BB")
        print("━" * 35)

        # 每 10 手显示统计
        if hand_num % 10 == 0:
            show_stats(hand_num, player_wins, bot_wins, player_stack, bot_stack, STARTING_STACK)

        # 检查游戏结束
        if player_stack < BB_AMOUNT:
            show_game_over("Bot", hand_num)
            break
        if bot_stack < BB_AMOUNT:
            show_game_over("你", hand_num)
            break

        # 下一手位置交换
        player_is_btn = not player_is_btn

        try:
            raw = input("\n按 Enter 继续下一手... (或输入 'q' 退出, 's' 统计, 'thinking' 切换思考显示) ")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = raw.strip().lower()
        if cmd in ("q", "quit"):
            break
        elif cmd in ("s", "status"):
            show_stats(hand_num, player_wins, bot_wins, player_stack, bot_stack, STARTING_STACK)
        elif cmd == "thinking":
            show_thinking = not show_thinking
            state = "开启" if show_thinking else "关闭"
            print(f"  Bot 思考过程已{state}。")

    # 退出时显示最终统计
    show_stats(hand_num, player_wins, bot_wins, player_stack, bot_stack, STARTING_STACK)
    print("感谢游玩！再见！")


if __name__ == "__main__":
    main()
