from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_fly_opponent import MatchupResult
from experiment_fly_maxpower_sweep import (
    CHECKPOINT_INTERVAL,
    EVALUATION_BATTLES,
    evaluate_checkpoint,
    train_with_checkpoints,
)


class MaxPowerCheckpointSweepTest(unittest.TestCase):
    def test_saves_after_each_tenth_existing_update_for_exactly_100_battles(self) -> None:
        rollout_seeds: list[int] = []
        saved: list[tuple[Path, int]] = []
        update_count = 0

        def rollout(_environment: object, _policy: object, *, seed: int) -> object:
            rollout_seeds.append(seed)
            return SimpleNamespace(
                outcome="win",
                total_reward=1.0,
                turns=(object(), object()),
                illegal_action_attempted=False,
            )

        def update(_policy: object, _rollout: object) -> None:
            nonlocal update_count
            update_count += 1

        def save(path: Path) -> None:
            saved.append((path, update_count))

        policy = SimpleNamespace(actor_critic=SimpleNamespace(save_weights=save))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch("experiment_fly_maxpower_sweep.run_one_battle", side_effect=rollout),
                patch("experiment_fly_maxpower_sweep.update_once", side_effect=update),
            ):
                training, checkpoints = train_with_checkpoints(
                    policy, object, output_dir=output_dir, seed=7
                )

        self.assertEqual(training.battles, 100)
        self.assertEqual(training.updates, 100)
        self.assertEqual(update_count, 100)
        self.assertEqual(rollout_seeds, list(range(7, 107)))
        self.assertEqual(len(checkpoints), 10)
        self.assertEqual(
            [path.name for path in checkpoints],
            [f"battle_{battle:03d}.pt" for battle in range(10, 101, 10)],
        )
        self.assertEqual([count for _, count in saved], list(range(10, 101, 10)))
        self.assertEqual(CHECKPOINT_INTERVAL, 10)
        self.assertEqual(EVALUATION_BATTLES, 50)

    def test_evaluates_checkpoint_read_only_for_50_battles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "battle_010.pt"
            checkpoint.write_bytes(b"fixed checkpoint")
            environment = SimpleNamespace(close=lambda: None)
            result = MatchupResult(
                name="max-power",
                wins=20,
                losses=30,
                ties=0,
                rewards=(1.0,) * EVALUATION_BATTLES,
                turns=(25,) * EVALUATION_BATTLES,
                fly_illegal_actions=0,
                opposing_illegal_actions=0,
            )
            with (
                patch("experiment_fly_maxpower_sweep.ShowdownEnv") as showdown,
                patch("experiment_fly_maxpower_sweep.SingleAgentWrapper", return_value=environment),
                patch("experiment_fly_maxpower_sweep.make_opponent") as make_opponent,
                patch("experiment_fly_maxpower_sweep.scripted_action", return_value=object()),
                patch("experiment_fly_maxpower_sweep.run_matchup", return_value=result) as run_matchup,
                patch("experiment_fly_maxpower_sweep.secrets.token_hex", return_value="abcdef1234"),
            ):
                actual = evaluate_checkpoint(checkpoint, object(), seed=7)

            self.assertIs(actual, result)
            self.assertEqual(checkpoint.read_bytes(), b"fixed checkpoint")
            self.assertEqual(run_matchup.call_args.kwargs, {"battles": 50, "seed": 7})
            self.assertEqual(make_opponent.call_args_list[0].args, ("fly",))
            self.assertEqual(make_opponent.call_args_list[0].kwargs["fly_checkpoint"], checkpoint)
            self.assertEqual(make_opponent.call_args_list[1].args, ("max-power",))
            kwargs = showdown.call_args.kwargs
            self.assertEqual(kwargs["account_configuration1"].username, "fsabcdef1234a")
            self.assertEqual(kwargs["account_configuration2"].username, "fsabcdef1234b")


if __name__ == "__main__":
    unittest.main()
