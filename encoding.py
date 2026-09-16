"""Stable battle-state encoding shared by every policy implementation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from poke_env.battle import Battle, Move, Pokemon, PokemonType
from poke_env.battle.move_category import MoveCategory


TEAM_SIZE = 6
MOVES_PER_POKEMON = 4
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_NAMES = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
POKEMON_TYPES = tuple(PokemonType)

# Six general battle values, ten binary battle flags, two teams, two active
# Pokemon, and four move descriptions. Keep this expression explicit so a future
# policy can depend on one public input dimension.
GLOBAL_FEATURES = 16
TEAM_FEATURES = 2 * TEAM_SIZE * 6
ACTIVE_FEATURES = 2 * (len(BOOST_NAMES) + len(STAT_NAMES) + len(POKEMON_TYPES))
MOVE_FEATURES = MOVES_PER_POKEMON * 8
OBSERVATION_DIM = GLOBAL_FEATURES + TEAM_FEATURES + ACTIVE_FEATURES + MOVE_FEATURES


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _team_features(team: Iterable[Pokemon]) -> list[float]:
    features: list[float] = []
    pokemon = list(team)[:TEAM_SIZE]
    for mon in pokemon:
        features.extend(
            (
                1.0,
                float(mon.active),
                _clip(mon.current_hp_fraction, 0.0, 1.0),
                float(mon.fainted),
                float(mon.status is not None),
                _clip(mon.level / 100.0, 0.0, 1.0),
            )
        )
    features.extend([0.0] * ((TEAM_SIZE - len(pokemon)) * 6))
    return features


def _active_features(mon: Pokemon | None) -> list[float]:
    size = len(BOOST_NAMES) + len(STAT_NAMES) + len(POKEMON_TYPES)
    if mon is None:
        return [0.0] * size

    features = [_clip(mon.boosts.get(name, 0) / 6.0) for name in BOOST_NAMES]
    features.extend(
        _clip(mon.base_stats.get(name, 0) / 255.0, 0.0, 1.0)
        for name in STAT_NAMES
    )
    mon_types = set(mon.types)
    features.extend(float(pokemon_type in mon_types) for pokemon_type in POKEMON_TYPES)
    return features


def _move_features(move: Move, user: Pokemon, target: Pokemon | None) -> list[float]:
    category = move.category
    accuracy = move.accuracy
    if accuracy is True or accuracy is None:
        accuracy_value = 1.0
    else:
        accuracy_value = _clip(float(accuracy), 0.0, 1.0)

    effectiveness = 1.0
    if target is not None and move.type is not None:
        effectiveness = target.damage_multiplier(move.type)

    return [
        _clip(move.base_power / 250.0, 0.0, 1.0),
        accuracy_value,
        _clip(move.priority / 7.0),
        float(category is MoveCategory.PHYSICAL),
        float(category is MoveCategory.SPECIAL),
        float(category is MoveCategory.STATUS),
        float(move.type in user.types),
        _clip(effectiveness / 4.0, 0.0, 1.0),
    ]


def _moves_features(user: Pokemon | None, target: Pokemon | None) -> list[float]:
    features: list[float] = []
    moves = list(user.moves.values())[:MOVES_PER_POKEMON] if user is not None else []
    for move in moves:
        features.extend(_move_features(move, user, target))
    features.extend([0.0] * ((MOVES_PER_POKEMON - len(moves)) * 8))
    return features


def embed_battle(battle: Battle) -> np.ndarray:
    """Convert a partially observed singles battle into a fixed float vector."""

    own_team = list(battle.team.values())
    opponent_team = list(battle.opponent_team.values())
    own_alive = sum(not mon.fainted for mon in own_team)
    opponent_alive = sum(not mon.fainted for mon in opponent_team)

    features = [
        _clip(battle.turn / 100.0, 0.0, 1.0),
        own_alive / TEAM_SIZE,
        opponent_alive / TEAM_SIZE,
        len(opponent_team) / TEAM_SIZE,
        sum(mon.current_hp_fraction for mon in own_team) / TEAM_SIZE,
        sum(mon.current_hp_fraction for mon in opponent_team) / TEAM_SIZE,
        float(battle.trapped),
        float(bool(battle.force_switch)),
        float(battle.can_mega_evolve),
        float(battle.can_z_move),
        float(battle.can_dynamax),
        float(battle.can_tera),
        float(bool(battle.won)),
        float(bool(battle.lost)),
        float(battle.finished),
        1.0,
    ]
    features.extend(_team_features(own_team))
    features.extend(_team_features(opponent_team))
    features.extend(_active_features(battle.active_pokemon))
    features.extend(_active_features(battle.opponent_active_pokemon))
    features.extend(_moves_features(battle.active_pokemon, battle.opponent_active_pokemon))

    observation = np.asarray(features, dtype=np.float32)
    if observation.shape != (OBSERVATION_DIM,):
        raise RuntimeError(
            f"Encoder produced {observation.size} values; expected {OBSERVATION_DIM}"
        )
    return observation
