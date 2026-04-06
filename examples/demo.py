"""Demo: showcasing poker bot capabilities."""

from src.card import cards_from_str, card_from_str
from src.position import Position
from src.bot import PokerBot, GameState
from src.preflop import PreflopEngine
from src.postflop import PostflopEngine
from src.hand_analysis import classify_hand
from src.board_analysis import analyze_board, get_board_texture_description
from src.barrel_plan import create_barrel_plan
from src.sizing import calculate_sizing
from src.opponent import VillainProfile
from src.icm import calculate_icm, icm_pressure
from src.weighted_range import build_range_combos
from src.ranges import get_rfi_range
from src.multiway import adjust_for_multiway


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_preflop_aa_utg():
    separator("1. Preflop: UTG opens with AA")
    engine = PreflopEngine()
    hand = tuple(cards_from_str("AsAh"))
    decision = engine.decide(
        hero_hand=hand, position=Position.UTG,
        action_history=[], pot=1.5, to_call=0.0,
        hero_stack=100.0, num_players=6, villain_profiles={},
    )
    print(f"Hand: AA | Position: UTG")
    print(f"Action: {decision.action} {decision.amount:.1f}BB")
    print(f"Reasoning: {decision.reasoning}")


def demo_preflop_btn_3bet():
    separator("2. Preflop: BTN 3bets vs CO open")
    engine = PreflopEngine()
    hand = tuple(cards_from_str("KsKh"))
    action_history = [{"position": Position.CO, "action": "raise", "amount": 2.5}]
    decision = engine.decide(
        hero_hand=hand, position=Position.BTN,
        action_history=action_history, pot=4.0, to_call=2.5,
        hero_stack=100.0, num_players=6, villain_profiles={},
    )
    print(f"Hand: KK | Position: BTN | CO opened 2.5BB")
    print(f"Action: {decision.action} {decision.amount:.1f}BB")
    print(f"Reasoning: {decision.reasoning}")


def demo_postflop_cbet_dry():
    separator("3. Postflop: C-bet decision on dry board (TPTK)")
    hand = tuple(cards_from_str("AhKc"))
    board = cards_from_str("As2d7h")
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.UTG), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    engine = PostflopEngine()
    decision = engine.decide(
        hero_hand=hand, board=board, pot=10.0, to_call=0.0,
        hero_stack=90.0, villain_stack=90.0, position=Position.BTN,
        street="flop", villain_profile=None,
        hero_range=hero_range, villain_range=villain_range,
        barrel_plan=None, is_multiway=False,
    )
    texture = analyze_board(board)
    print(f"Hand: AhKc | Board: As 2d 7h")
    print(f"Board texture: {get_board_texture_description(texture)}")
    hs = classify_hand(hand, board)
    print(f"Hand strength: {hs.made_hand.name}")
    print(f"Action: {decision.action} {decision.amount:.1f}")
    print(f"Reasoning: {decision.reasoning}")


def demo_postflop_checkraise_bluff():
    separator("4. Postflop: Check-raise bluff (nut flush draw)")
    hand = tuple(cards_from_str("AsTs"))
    board = cards_from_str("2s5s9h")
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.CO), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BB), dead)
    engine = PostflopEngine()
    # Villain bet 6 into 10, hero faces bet
    decision = engine.decide(
        hero_hand=hand, board=board, pot=10.0, to_call=6.0,
        hero_stack=90.0, villain_stack=90.0, position=Position.BB,
        street="flop", villain_profile=None,
        hero_range=hero_range, villain_range=villain_range,
        barrel_plan=None, is_multiway=False,
    )
    hs = classify_hand(hand, board)
    print(f"Hand: AsTs | Board: 2s 5s 9h | Villain bet 6 into pot 10")
    print(f"Hand strength: {hs.made_hand.name} | Draw: {hs.draw.name}")
    print(f"Action: {decision.action} {decision.amount:.1f}")
    print(f"Reasoning: {decision.reasoning}")


def demo_multiway():
    separator("5. Multiway pot strategy")
    hand = tuple(cards_from_str("JhTh"))
    board = cards_from_str("Jd5c2h")
    hs = classify_hand(hand, board)
    texture = analyze_board(board)
    adj = adjust_for_multiway(hs, 3, texture)
    print(f"Hand: JhTh | Board: Jd 5c 2h | 3-way pot")
    print(f"Hand strength: {hs.made_hand.name}")
    print(f"Bet frequency multiplier: {adj.bet_frequency_multiplier:.2f}")
    print(f"Value threshold adjustment: +{adj.value_threshold_adjustment}")
    print(f"Bluffing allowed: {adj.bluff_allowed}")


def demo_icm_bubble():
    separator("6. ICM bubble scenario")
    stacks = [8000.0, 7000.0, 5000.0, 2000.0, 1500.0]
    payouts = [10000.0, 6000.0, 4000.0, 2500.0]  # top 4 paid
    equities = calculate_icm(stacks, payouts)
    print(f"Players: {len(stacks)} | Paid spots: {len(payouts)}")
    for i, (stack, eq) in enumerate(zip(stacks, equities)):
        pressure = icm_pressure(stack, stacks, payouts, i)
        print(f"  Player {i+1}: stack={stack:.0f} ICM_equity=${eq:.0f} pressure={pressure:.2f}")


def demo_opponent_profiling():
    separator("7. Opponent profiling")
    profile = VillainProfile()
    # Simulate observed actions
    for _ in range(15):
        profile.update_action("preflop", "fold", Position.CO, 0.0, 3.0)
    for _ in range(5):
        profile.update_action("preflop", "raise", Position.BTN, 2.5, 1.5)
    for _ in range(8):
        profile.update_action("flop", "bet", Position.BTN, 5.0, 10.0)
    for _ in range(3):
        profile.update_action("flop", "check", Position.BTN, 0.0, 10.0)

    print(f"VPIP: {profile.stats.vpip:.1%} | PFR: {profile.stats.pfr:.1%}")
    print(f"Cbet flop: {profile.stats.cbet_flop:.1%}")
    print(f"Is fish: {profile.is_fish()} | Is nit: {profile.is_nit()} | Is TAG: {profile.is_tag()}")
    suggestion = profile.get_exploit_suggestion("flop", "cbet_opportunity")
    print(f"Exploit: {suggestion}")


def demo_barrel_planning():
    separator("8. Barrel planning example")
    hand = tuple(cards_from_str("AhKh"))
    board = cards_from_str("Ac7d2s")
    hs = classify_hand(hand, board)
    texture = analyze_board(board)
    dead = list(hand) + board
    villain_range = build_range_combos(get_rfi_range(Position.UTG), dead)
    hero_range = build_range_combos(get_rfi_range(Position.BTN), dead)
    from src.board_analysis import analyze_range_advantage
    range_adv = analyze_range_advantage(hero_range, villain_range, board)
    plan = create_barrel_plan(hs, texture, range_adv, "ip", spr=8.0)
    print(f"Hand: AhKh | Board: Ac 7d 2s | Position: IP | SPR: 8.0")
    print(f"Hand: {hs.made_hand.name} | Board: {get_board_texture_description(texture)}")
    print(f"Barrel plan:")
    print(f"  Flop:  {plan.flop_action}")
    print(f"  Turn:  {plan.turn_action}")
    print(f"  River: {plan.river_action}")
    if plan.value_line:
        print(f"  Value line: {plan.value_line.value}")
    if plan.bluff_line:
        print(f"  Bluff line: {plan.bluff_line.value}")
    print(f"  Continue runouts: {plan.continue_runouts}")
    print(f"  Give-up runouts:  {plan.give_up_runouts}")


if __name__ == "__main__":
    demo_preflop_aa_utg()
    demo_preflop_btn_3bet()
    demo_postflop_cbet_dry()
    demo_postflop_checkraise_bluff()
    demo_multiway()
    demo_icm_bubble()
    demo_opponent_profiling()
    demo_barrel_planning()
    print("\n✅ Demo completed successfully.")
