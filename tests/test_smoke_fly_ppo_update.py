from __future__ import annotations

import unittest

import numpy as np
import torch

from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns, update_once
from smoke_fly_representation_diversity import observation_profiles
from smoke_fly_rollout import RolloutResult, TurnRecord


def recorded_rollout(policy: FlyCNSPPOPolicy) -> RolloutResult:
    observations = list(observation_profiles().values())[:3]
    masks = np.zeros((3, ACTION_COUNT), dtype=np.int8)
    masks[:, [2, 7, 11]] = 1
    actions = np.asarray([2, 7, 11], dtype=np.int64)
    old = policy.evaluate_batch(observations, masks, actions)
    turns = tuple(
        TurnRecord(
            observation=observation.copy(),
            action_mask=mask.copy(),
            selected_action=int(action),
            log_probability=float(log_probability),
            state_value=float(value),
            reward=reward,
            done=index == 2,
        )
        for index, (observation, mask, action, log_probability, value, reward) in enumerate(
            zip(
                observations,
                masks,
                actions,
                old.action_log_probabilities,
                old.state_values,
                (0.4, -0.2, 2.0),
                strict=True,
            )
        )
    )
    return RolloutResult(turns, 2.2, "win", False, True)


class FlyPPOUpdateSmokeTest(unittest.TestCase):
    def test_gae_uses_recorded_values_and_terminal_reward(self) -> None:
        policy = FlyCNSPPOPolicy()
        rollout = recorded_rollout(policy)
        advantages, returns = advantages_and_returns(rollout, gamma=1.0, gae_lambda=1.0)

        np.testing.assert_allclose(returns, [2.2, 1.8, 2.0], rtol=0, atol=1e-5)
        values = np.asarray([turn.state_value for turn in rollout.turns])
        np.testing.assert_allclose(advantages, returns - values, rtol=0, atol=1e-5)

    def test_exactly_one_update_changes_heads_but_not_fly_output(self) -> None:
        torch.manual_seed(7)
        policy = FlyCNSPPOPolicy()
        rollout = recorded_rollout(policy)
        before_activity = policy._descending_activity(rollout.turns[0].observation)
        actor_before = [item.detach().clone() for item in policy.actor_critic.policy_head.parameters()]
        critic_before = [item.detach().clone() for item in policy.actor_critic.value_head.parameters()]

        result = update_once(policy, rollout)

        self.assertTrue(result.actor_changed)
        self.assertTrue(result.critic_changed)
        self.assertTrue(result.fly_output_unchanged)
        self.assertTrue(np.array_equal(
            before_activity, policy._descending_activity(rollout.turns[0].observation)
        ))
        self.assertTrue(any(
            not torch.equal(item, before)
            for item, before in zip(policy.actor_critic.policy_head.parameters(), actor_before, strict=True)
        ))
        self.assertTrue(any(
            not torch.equal(item, before)
            for item, before in zip(policy.actor_critic.value_head.parameters(), critic_before, strict=True)
        ))
        self.assertTrue(np.isfinite([
            result.policy_loss, result.value_loss, result.entropy, result.total_loss
        ]).all())


if __name__ == "__main__":
    unittest.main()
