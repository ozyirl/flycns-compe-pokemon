from __future__ import annotations

import unittest

from diagnose_fly_reward_credit import (
    BattleState,
    COMPONENTS,
    EVENTS,
    MonState,
    StepObservation,
    direct_event_reward,
    observed_events,
    reward_components,
    summarize,
)


def state(
    *, own_hp: float = 1.0, opponent_hp: float = 1.0,
    own_fainted: bool = False, opponent_fainted: bool = False,
    own_status: str | None = None, opponent_status: str | None = None,
    active: str = "own", won: bool = False, lost: bool = False,
) -> BattleState:
    return BattleState(
        own={"own": MonState(own_hp, own_fainted, own_status, active == "own")},
        opponent={"opp": MonState(opponent_hp, opponent_fainted, opponent_status, True)},
        won=won, lost=lost, active_own=active,
    )


class RewardCreditTest(unittest.TestCase):
    def test_exact_existing_reward_terms_and_events(self) -> None:
        before = state()
        after = state(own_hp=0.6, opponent_hp=0.0, opponent_fainted=True,
                      own_status="PAR", won=True)
        previous = reward_components(before)
        current = reward_components(after)
        delta = {key: current[key] - previous[key] for key in current}
        self.assertAlmostEqual(delta["own_hp"], -0.2)
        self.assertAlmostEqual(delta["opponent_hp"], 0.5)
        self.assertAlmostEqual(delta["opponent_faint"], 2.0)
        self.assertAlmostEqual(delta["own_status"], -0.1)
        self.assertAlmostEqual(delta["victory"], 10.0)
        self.assertAlmostEqual(sum(delta.values()), 12.2)
        events = observed_events(StepObservation(before, after, 6, "move", 12.2))
        self.assertTrue(events["damage_dealt"])
        self.assertTrue(events["damage_received"])
        self.assertTrue(events["opponent_fainted"])
        self.assertTrue(events["status_received"])
        self.assertTrue(events["final_win"])
        self.assertFalse(events["successful_switch"])
        direct = direct_event_reward(StepObservation(before, after, 6, "move", 12.2))
        self.assertAlmostEqual(direct["damage_dealt"], 0.5)
        self.assertAlmostEqual(direct["damage_received"], -0.2)
        self.assertAlmostEqual(direct["opponent_fainted"], 2.0)
        self.assertAlmostEqual(direct["final_win"], 10.0)

    def test_switch_is_observed_but_has_no_direct_reward_term(self) -> None:
        before = BattleState(
            own={"a": MonState(1, False, None, True), "b": MonState(1, False, None, False)},
            opponent={"x": MonState(1, False, None, True)}, won=False, lost=False,
            active_own="a",
        )
        after = BattleState(
            own={"a": MonState(1, False, None, False), "b": MonState(1, False, None, True)},
            opponent=before.opponent, won=False, lost=False, active_own="b",
        )
        events = observed_events(StepObservation(before, after, 1, "switch", 0))
        self.assertTrue(events["successful_switch"])
        self.assertFalse(observed_events(StepObservation(before, after, 0, "wait/default", 0))["successful_switch"])
        self.assertEqual(sum(reward_components(after).values()) - sum(reward_components(before).values()), 0)

    def test_summary_keeps_overlapping_event_rewards_and_battle_boundaries(self) -> None:
        flags = {name: False for name in (
            "damage_dealt", "damage_received", "opponent_fainted", "own_fainted",
            "successful_switch", "status_inflicted", "status_received", "final_win", "final_loss",
        )}
        rows = []
        for battle, turn, advantage in ((1, 1, 1.0), (1, 2, 3.0), (2, 1, -5.0)):
            events = flags.copy()
            if battle == 1 and turn == 2:
                events["damage_dealt"] = events["opponent_fainted"] = True
            rows.append({"battle": battle, "turn": turn, "outcome": "win" if battle == 1 else "loss",
                         "reward": float(2 if turn == 2 else 0), "ppo_advantage": advantage,
                         "events": events, "reward_components": {
                             name: float(2 if turn == 2 else 0) if name == "victory" else 0.0
                             for name in COMPONENTS
                         }, "direct_event_reward_terms": {name: 0.0 for name in EVENTS}})
        result = summarize(rows)
        event = result["events_observed"]["damage_dealt"]
        self.assertEqual(result["battles"], 2)
        self.assertEqual(event["reward_sum_on_event_turns_overlapping"], 2.0)
        self.assertEqual(result["events_observed"]["opponent_fainted"]["reward_sum_on_event_turns_overlapping"], 2.0)
        self.assertEqual(event["mean_advantage_previous_turn"], 1.0)
        self.assertEqual(event["mean_advantage_event_turn"], 3.0)
        self.assertIsNone(event["mean_advantage_next_turn"])


if __name__ == "__main__":
    unittest.main()
