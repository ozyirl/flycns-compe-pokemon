from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from poke_env.battle import Move, Pokemon

from encoding import (
    MATCHUP_FEATURE_SIZE,
    MATCHUP_FEATURES,
    MATCHUP_OBSERVATION_DIM,
    OBSERVATION_DIM,
    embed_matchups,
)


def _pokemon(species: str, *moves: str) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    for move in moves:
        mon._moves[move] = Move(move, gen=9)
    return mon


class MatchupEncodingTest(unittest.TestCase):
    def test_dimensions_append_one_block_per_team_slot(self) -> None:
        self.assertEqual(MATCHUP_FEATURES, 6 * MATCHUP_FEATURE_SIZE)
        self.assertEqual(MATCHUP_OBSERVATION_DIM, OBSERVATION_DIM + MATCHUP_FEATURES)

    def test_scores_each_slot_against_the_opponent_active(self) -> None:
        garchomp = _pokemon("garchomp", "earthquake", "swordsdance")
        blissey = _pokemon("blissey", "softboiled")
        heatran = _pokemon("heatran", "flamethrower")
        battle = SimpleNamespace(
            team={"p1: Garchomp": garchomp, "p1: Blissey": blissey},
            opponent_active_pokemon=heatran,
        )

        matchups = embed_matchups(battle).reshape(6, MATCHUP_FEATURE_SIZE)

        # Ground hits Fire/Steel 4x; Heatran's best STAB is neutral; Garchomp is faster;
        # STAB Earthquake is 100 x 1.5 x 4; resisted STAB Flamethrower is 90 x 1.5 x 0.5.
        np.testing.assert_allclose(matchups[0], [1.0, 0.25, 1.0, 1.0, 67.5 / 600.0])
        # Normal is resisted by Steel; Blissey is slower and has no damaging move.
        np.testing.assert_allclose(matchups[1], [0.125, 0.25, -1.0, 0.0, 135.0 / 600.0])
        np.testing.assert_array_equal(matchups[2:], 0.0)

    def test_zeros_without_an_opponent_active(self) -> None:
        battle = SimpleNamespace(team={"p1: Garchomp": _pokemon("garchomp")}, opponent_active_pokemon=None)

        np.testing.assert_array_equal(embed_matchups(battle), np.zeros(MATCHUP_FEATURES, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
