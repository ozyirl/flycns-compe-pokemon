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

# Optional block appended after the stable observation: how each of the trainer's
# team slots matches up against the opponent's active Pokemon. The Fly CNS input
# and older checkpoints keep reading only the first OBSERVATION_DIM values.
MATCHUP_FEATURE_SIZE = 5
MATCHUP_FEATURES = TEAM_SIZE * MATCHUP_FEATURE_SIZE
MATCHUP_OBSERVATION_DIM = OBSERVATION_DIM + MATCHUP_FEATURES
# Base power x accuracy x STAB x effectiveness of a strong super-effective STAB move.
EXPECTED_POWER_SCALE = 600.0


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


def _expected_power(move: Move, user: Pokemon, target: Pokemon) -> float:
    if move.category is MoveCategory.STATUS or move.type is None:
        return 0.0
    accuracy = 1.0 if move.accuracy is True or move.accuracy is None else float(move.accuracy)
    stab = 1.5 if move.type in user.types else 1.0
    return move.base_power * accuracy * stab * target.damage_multiplier(move.type)


def _best_type_effectiveness(attacker: Pokemon, defender: Pokemon) -> float:
    return max((defender.damage_multiplier(t) for t in attacker.types if t is not None), default=1.0)


def _slot_matchup(mon: Pokemon, opponent: Pokemon) -> list[float]:
    speed = np.sign(mon.base_stats.get("spe", 0) - opponent.base_stats.get("spe", 0))
    best_move = max((_expected_power(move, mon, opponent) for move in mon.moves.values()), default=0.0)
    threat = max((_expected_power(move, opponent, mon) for move in opponent.moves.values()), default=0.0)
    return [
        _clip(_best_type_effectiveness(mon, opponent) / 4.0, 0.0, 1.0),
        _clip(_best_type_effectiveness(opponent, mon) / 4.0, 0.0, 1.0),
        float(speed),
        _clip(best_move / EXPECTED_POWER_SCALE, 0.0, 1.0),
        _clip(threat / EXPECTED_POWER_SCALE, 0.0, 1.0),
    ]


def embed_matchups(battle: Battle) -> np.ndarray:
    """Score each own team slot against the opponent's active Pokemon.

    Per slot: best STAB-type effectiveness dealt and taken (/4), base-speed
    comparison (-1, 0, 1), and the expected power of the slot's best known move
    and of the opponent's best revealed move against it. Zeros when there is no
    opponent active Pokemon or no team member in the slot.
    """

    features: list[float] = []
    opponent = battle.opponent_active_pokemon
    pokemon = list(battle.team.values())[:TEAM_SIZE]
    for mon in pokemon:
        features.extend(
            _slot_matchup(mon, opponent) if opponent is not None else [0.0] * MATCHUP_FEATURE_SIZE
        )
    features.extend([0.0] * ((TEAM_SIZE - len(pokemon)) * MATCHUP_FEATURE_SIZE))
    return np.asarray(features, dtype=np.float32)


def embed_battle_with_matchups(battle: Battle) -> np.ndarray:
    """The stable observation followed by :func:`embed_matchups`."""

    return np.concatenate([embed_battle(battle), embed_matchups(battle)])


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
