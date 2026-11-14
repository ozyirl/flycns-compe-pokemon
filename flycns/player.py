"""Checkpoint-backed, inference-only Fly CNS opponent for the singles trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from encoding import OBSERVATION_DIM, embed_battle, embed_matchups
from flycns.hybrid_actor_critic import load_actor_critic
from flycns.ppo_policy import FlyCNSPPOPolicy, PolicyDecision


def load_frozen_policy(checkpoint_path: Path) -> FlyCNSPPOPolicy:
    """Load a checkpoint for inference only, with every parameter frozen."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fly CNS checkpoint not found: {checkpoint_path}")
    actor_critic = load_actor_critic(checkpoint_path)
    actor_critic.eval()
    actor_critic.requires_grad_(False)
    return FlyCNSPPOPolicy(actor_critic=actor_critic)


def decide(
    policy: FlyCNSPPOPolicy, battle: Battle, *, deterministic: bool = True
) -> tuple[BattleOrder, PolicyDecision]:
    """Encode ``battle`` the way training did and convert the policy's choice to an order."""

    observation = embed_battle(battle)
    if policy.actor_critic.observation_size > OBSERVATION_DIM:
        observation = np.concatenate([observation, embed_matchups(battle)])
    action_mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
    decision = policy.predict(observation, action_mask, deterministic=deterministic)
    action = decision.selected_action_index
    if not 0 <= action < action_mask.size or action_mask[action] != 1:
        raise ValueError(f"Fly CNS selected illegal action {action}")
    return SinglesEnv.action_to_order(np.int64(action), battle, strict=True), decision


class FlyCNSPlayer(Player):
    """Choose legal Showdown orders with frozen Fly-only or hybrid actor-critic heads."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        deterministic: bool = True,
        **player_kwargs: Any,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.policy = load_frozen_policy(self.checkpoint_path)
        self.deterministic = deterministic
        self.choices_made = 0
        self.illegal_action_count = 0
        super().__init__(**player_kwargs)

    def choose_move(self, battle: Battle) -> BattleOrder:
        try:
            order, _ = decide(self.policy, battle, deterministic=self.deterministic)
        except ValueError:
            self.illegal_action_count += 1
            raise
        self.choices_made += 1
        return order
