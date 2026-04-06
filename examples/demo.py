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
from src.range_updater import RangeUpdater
from src.outs_calculator import OutsCalculator, format_outs_summary
from src.sizing_tell import SizingTellInterpreter


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


def demo_range_updater():
    separator("9. Range Updater – narrowing villain range based on actions")
    board = cards_from_str("AcKd7h")
    tex = analyze_board(board)
    dead = board
    original = build_range_combos(get_rfi_range(Position.CO), dead)

    updater = RangeUpdater()
    print(f"Board: Ac Kd 7h  |  Starting villain range: {len(original)} combos")

    # After villain checks
    after_check = updater.update_range_after_action(
        original, board, "check", 0.0, "flop", tex, is_ip=True
    )
    print(f"After villain CHECK:         {len(after_check)} combos (strong hands removed)")

    # After small bet
    after_small = updater.update_range_after_action(
        original, board, "bet", 0.30, "flop", tex, is_ip=True
    )
    print(f"After villain BET 30%:       {len(after_small)} combos (merged range)")

    # After large bet
    after_large = updater.update_range_after_action(
        original, board, "bet", 0.85, "flop", tex, is_ip=True
    )
    print(f"After villain BET 85%:       {len(after_large)} combos (polarized – strong+bluffs)")

    # After raise
    after_raise = updater.update_range_after_action(
        original, board, "raise", 2.5, "flop", tex, is_ip=True
    )
    print(f"After villain RAISE 2.5x:    {len(after_raise)} combos (very strong / some bluffs)")

    # After all-in
    after_allin = updater.update_range_after_action(
        original, board, "all_in", 0.0, "flop", tex, is_ip=True
    )
    print(f"After villain ALL-IN:        {len(after_allin)} combos (nuts + minimal bluffs)")

    # Monotone board all-in
    board_mono = cards_from_str("KsQsJs")
    tex_mono = analyze_board(board_mono)
    original_mono = build_range_combos(get_rfi_range(Position.CO), board_mono)
    after_mono_allin = updater.update_range_after_action(
        original_mono, board_mono, "all_in", 0.0, "flop", tex_mono, is_ip=False
    )
    print(f"\nMonotone board (KsQsJs) villain ALL-IN: {len(after_mono_allin)} combos")
    print("  (flush-heavy range – bluffing monotone all-in is rare)")


def demo_outs_calculator():
    separator("10. Precise Outs Calculator")
    calc = OutsCalculator()

    # Scenario 1: Nut flush draw
    hole1 = tuple(cards_from_str("As9s"))
    board1 = cards_from_str("Ks7s2h")
    dead1 = list(hole1) + board1
    vrange1 = build_range_combos(get_rfi_range(Position.CO), dead1)
    analysis1 = calc.calculate_outs(hole1, board1, vrange1, simulations_per_out=100)
    print("Scenario A: Nut flush draw (As9s on Ks7s2h)")
    print(f"  {format_outs_summary(analysis1)}")
    if analysis1.best_out:
        print(f"  Best out: {analysis1.best_out.card} → {analysis1.best_out.improves_to} "
              f"(equity after: {analysis1.best_out.equity_after:.0%})")

    # Scenario 2: OESD
    hole2 = tuple(cards_from_str("JhTh"))
    board2 = cards_from_str("9s8d2c")
    dead2 = list(hole2) + board2
    vrange2 = build_range_combos(get_rfi_range(Position.CO), dead2)
    analysis2 = calc.calculate_outs(hole2, board2, vrange2, simulations_per_out=100)
    print("\nScenario B: OESD (JhTh on 9s8d2c)")
    print(f"  {format_outs_summary(analysis2)}")

    # Scenario 3: Set vs flush board
    hole3 = tuple(cards_from_str("KhKd"))
    board3 = cards_from_str("Ks9s5s")
    dead3 = list(hole3) + board3
    vrange3 = build_range_combos(get_rfi_range(Position.CO), dead3)
    analysis3 = calc.calculate_outs(hole3, board3, vrange3, simulations_per_out=100)
    print("\nScenario C: Set of Kings vs monotone board (KhKd on Ks9s5s)")
    print(f"  {format_outs_summary(analysis3)}")
    print(f"  Note: full house outs may be dirty if flush is already out there")

    # Scenario 4: Combo draw
    hole4 = tuple(cards_from_str("JsTs"))
    board4 = cards_from_str("9s8s2h")
    dead4 = list(hole4) + board4
    vrange4 = build_range_combos(get_rfi_range(Position.CO), dead4)
    analysis4 = calc.calculate_outs(hole4, board4, vrange4, simulations_per_out=100)
    print("\nScenario D: Combo draw (JsTs on 9s8s2h – flush + OESD)")
    print(f"  {format_outs_summary(analysis4)}")
    print(f"  True equity ≈ {analysis4.true_equity:.0%} → strong semi-bluff candidate")


def demo_sizing_tell():
    separator("11. Sizing Tell Interpreter – infer range from bet size")
    interp = SizingTellInterpreter()

    board_dry = cards_from_str("Ac7d2h")
    tex_dry = analyze_board(board_dry)
    board_mono = cards_from_str("KsQsJs")
    tex_mono = analyze_board(board_mono)

    scenarios = [
        ("Dry board, 25% bet", "bet", 0.25, "flop", tex_dry),
        ("Dry board, 50% bet", "bet", 0.50, "flop", tex_dry),
        ("Dry board, 85% bet", "bet", 0.85, "flop", tex_dry),
        ("Dry board, 130% overbet", "bet", 1.30, "flop", tex_dry),
        ("Monotone board, all-in", "all_in", 0.0, "flop", tex_mono),
        ("Dry board, river 120% overbet", "bet", 1.20, "river", tex_dry),
    ]

    for label, action, ratio, street, tex in scenarios:
        result = interp.interpret(action, ratio, street, tex)
        print(f"\n  {label}:")
        print(f"    Polarization:  {result.polarization}")
        print(f"    Value %:       {result.estimated_value_pct:.0%}")
        print(f"    Bluff %:       {result.estimated_bluff_pct:.0%}")
        print(f"    Description:   {result.description}")

    # Scenario: bot facing monotone board all-in
    separator("12. Bot decision: facing all-in on monotone board (KsQsJs)")
    bot = PokerBot(Position.BTN)
    hand = tuple(cards_from_str("AhAd"))  # pocket aces but no flush
    board = cards_from_str("KsQsJs")
    gs = GameState(
        hero_hand=hand,
        board=board,
        pot=50.0,
        to_call=100.0,       # villain all-in
        hero_stack=100.0,
        villain_stacks=[100.0],
        hero_position=Position.BTN,
        villain_positions=[Position.BB],
        street="flop",
        action_history=[{
            "actor": 1, "action": "all_in", "amount": 100.0,
            "pot": 50.0, "street": "flop", "is_ip": False,
        }],
    )
    result = bot.decide(gs)
    hand_str = classify_hand(hand, board)
    print(f"\nHand: AhAd | Board: KsQsJs (monotone)")
    print(f"Hand strength: {hand_str.made_hand.name}")
    print(f"Villain all-in (100BB into 50BB pot)")
    print(f"Bot action: {result['action']} {result['amount']:.1f}")
    print(f"Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    demo_preflop_aa_utg()
    demo_preflop_btn_3bet()
    demo_postflop_cbet_dry()
    demo_postflop_checkraise_bluff()
    demo_multiway()
    demo_icm_bubble()
    demo_opponent_profiling()
    demo_barrel_planning()
    demo_range_updater()
    demo_outs_calculator()
    demo_sizing_tell()
    print("\n✅ Demo completed successfully.")
