"""poke-env environment and single-agent training wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from poke_env.battle import Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from encoding import MATCHUP_OBSERVATION_DIM, OBSERVATION_DIM, embed_battle, embed_battle_with_matchups


DEFAULT_BATTLE_FORMAT = "gen9randombattle"


class ShowdownEnv(SinglesEnv):
    """Singles battle environment with the project's shared state and reward.

    ``include_matchups`` appends the per-slot matchup block after the stable
    observation, for policies that read it.
    """

    def __init__(self, *, include_matchups: bool = False, **kwargs: Any) -> None:
        self.include_matchups = include_matchups
        super().__init__(**kwargs)
        state_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MATCHUP_OBSERVATION_DIM if include_matchups else OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: state_space for agent in self.possible_agents
        }

    def embed_battle(self, battle: Battle) -> np.ndarray:
        return embed_battle_with_matchups(battle) if self.include_matchups else embed_battle(battle)

    def calc_reward(self, battle: Battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=0.5,
            status_value=0.1,
            victory_value=10.0,
        )


def make_opponent(
    kind: str,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    *,
    fly_checkpoint: str | Path | None = None,
) -> Player:
    """Build the scripted opponent used by the single-agent wrapper."""

    if kind == "fly":
        if fly_checkpoint is None:
            raise ValueError("Fly CNS opponent requires fly_checkpoint")
        from flycns.player import FlyCNSPlayer

        return FlyCNSPlayer(
            checkpoint_path=fly_checkpoint,
            battle_format=battle_format,
            start_listening=False,
        )
    opponents: dict[str, type[Player]] = {
        "random": RandomPlayer,
        "max-power": MaxBasePowerPlayer,
        "heuristics": SimpleHeuristicsPlayer,
    }
    try:
        opponent_class = opponents[kind]
    except KeyError as error:
        choices = ", ".join(sorted(opponents))
        raise ValueError(f"Unknown opponent {kind!r}; choose one of: {choices}") from error
    return opponent_class(battle_format=battle_format, start_listening=False)


def make_env(
    *,
    opponent: str = "random",
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    fly_checkpoint: str | Path | None = None,
) -> SingleAgentWrapper:
    """Create the Gymnasium environment consumed by Stable-Baselines3."""

    parallel_env = ShowdownEnv(battle_format=battle_format)
    return SingleAgentWrapper(
        parallel_env,
        make_opponent(
            opponent, battle_format=battle_format, fly_checkpoint=fly_checkpoint
        ),
    )
