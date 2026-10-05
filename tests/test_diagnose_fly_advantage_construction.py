from __future__ import annotations

import unittest

import numpy as np

from diagnose_fly_advantage_construction import (
    GAE_LAMBDA,
    GAMMA,
    advantages_normalized_before_policy_loss,
    select_representative_turns,
    trace_rollout,
)
from diagnose_fly_reward_credit import BattleState, MonState, StepObservation
from smoke_fly_ppo_update import advantages_and_returns
from smoke_fly_rollout import RolloutResult, TurnRecord


def battle_state(*, opponent_fainted: bool = False, own_fainted: bool = False, lost: bool = False) -> BattleState:
    return BattleState(
        own={"a": MonState(0.0 if own_fainted else 1.0, own_fainted, None, True)},
        opponent={"b": MonState(0.0 if opponent_fainted else 1.0, opponent_fainted, None, True)},
        won=False, lost=lost, active_own="a",
    )


def sample() -> tuple[RolloutResult, tuple[StepObservation, ...]]:
    observation = np.zeros(186, dtype=np.float32)
    mask = np.ones(26, dtype=np.int8)
    turns = (
        TurnRecord(observation, mask, 6, -1.0, 2.0, 5.0, False),
        TurnRecord(observation, mask, 6, -1.0, 3.0, -10.0, True),
    )
    rollout = RolloutResult(turns, -5.0, "loss", False, True)
    start = battle_state()
    knockout = battle_state(opponent_fainted=True)
    finish = battle_state(opponent_fainted=True, own_fainted=True, lost=True)
    steps = (
        StepObservation(start, knockout, 6, "move", 5.0),
        StepObservation(knockout, finish, 6, "move", -10.0),
    )
    return rollout, steps


class AdvantageConstructionTest(unittest.TestCase):
    def test_trace_matches_existing_gae_and_exposes_negative_knockout_advantage(self) -> None:
        rollout, steps = sample()
        battle = trace_rollout(rollout, steps, battle_index=4)
        advantages, returns = advantages_and_returns(rollout)
        first, last = battle.turns

        self.assertEqual((GAMMA, GAE_LAMBDA), (0.99, 0.95))
        self.assertEqual(first.events, ("opponent_fainted",))
        self.assertEqual(last.events, ("own_fainted", "final_loss"))
        self.assertEqual(first.next_value, 3.0)
        self.assertEqual(last.next_value, 0.0)
        self.assertAlmostEqual(first.td_delta, 5.0 + 0.99 * 3.0 - 2.0)
        self.assertAlmostEqual(last.td_delta, -10.0 - 3.0)
        self.assertAlmostEqual(first.future_gae_contribution, 0.99 * 0.95 * last.advantage)
        self.assertLess(first.advantage, 0.0)
        for index, row in enumerate(battle.turns):
            self.assertAlmostEqual(row.advantage, float(advantages[index]))
            self.assertAlmostEqual(row.computed_return, float(returns[index]))
            self.assertAlmostEqual(row.td_delta + row.future_gae_contribution, row.advantage, places=5)
        self.assertIs(select_representative_turns((battle,))["opponent_fainted"], first)

    def test_reports_raw_unnormalized_advantage_path(self) -> None:
        self.assertFalse(advantages_normalized_before_policy_loss())

    def test_rejects_unaligned_event_steps(self) -> None:
        rollout, steps = sample()
        with self.assertRaises(ValueError):
            trace_rollout(rollout, steps[:1], battle_index=1)
        changed = StepObservation(steps[0].before, steps[0].after, 7, "move", 5.0)
        with self.assertRaises(AssertionError):
            trace_rollout(rollout, (changed, steps[1]), battle_index=1)


if __name__ == "__main__":
    unittest.main()
