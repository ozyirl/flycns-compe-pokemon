"""Checkpoint-backed, inference-only Fly CNS opponent for the singles trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from encoding import embed_battle
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy


class FlyCNSPlayer(Player):
    """Choose legal Showdown orders with frozen Fly CNS actor-critic heads."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        deterministic: bool = True,
        **player_kwargs: Any,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Fly CNS checkpoint not found: {self.checkpoint_path}")
        actor_critic = FlyCNSActorCritic.load_weights(self.checkpoint_path)
        actor_critic.eval()
        actor_critic.requires_grad_(False)
        self.policy = FlyCNSPPOPolicy(actor_critic=actor_critic)
        self.deterministic = deterministic
        self.choices_made = 0
        self.illegal_action_count = 0
        super().__init__(**player_kwargs)

    def choose_move(self, battle: Battle) -> BattleOrder:
        observation = embed_battle(battle)
        action_mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        decision = self.policy.predict(
            observation, action_mask, deterministic=self.deterministic
        )
        action = decision.selected_action_index
        if not 0 <= action < action_mask.size or action_mask[action] != 1:
            self.illegal_action_count += 1
            raise ValueError(f"Fly CNS selected illegal action {action}")
        try:
            order = SinglesEnv.action_to_order(np.int64(action), battle, strict=True)
        except ValueError:
            self.illegal_action_count += 1
            raise
        self.choices_made += 1
        return order
