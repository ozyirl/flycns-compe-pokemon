"""Rebuild a poke-env battle from one player's Showdown protocol stream."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from poke_env.battle import Battle, Move, Pokemon
from poke_env.data import GenData
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder


LOGGER = logging.getLogger(__name__)


def replay_battle(protocol: Sequence[str], battle_format: str) -> tuple[Battle, dict[str, Any]]:
    """Feed ``protocol`` to a fresh battle the way a poke-env player handles server messages.

    The stream must be one player's view, in the order a Showdown server delivers it: each
    ``|request|`` line comes after the log lines it describes. The battle is returned with
    the last request applied, together with that request.
    """

    requests = [line for line in protocol if line.startswith("|request|") and len(line) > 9]
    if not requests:
        raise ValueError("protocol contains no request")
    request = json.loads(requests[-1][9:])
    role = request["side"]["id"]
    usernames = {
        parts[2]: parts[3]
        for parts in (line.split("|") for line in protocol if line.startswith("|player|"))
        if len(parts) > 3
    }
    if role not in usernames:
        raise ValueError(f"protocol has no |player| line for {role}")

    battle = Battle(
        battle_tag=f"battle-{battle_format}-replay",
        username=usernames[role],
        logger=LOGGER,
        gen=GenData.from_format(battle_format).gen,
    )
    for line in protocol:
        split_message = line.split("|")
        # Mirrors the per-line dispatch in poke-env's Player._handle_battle_message.
        if len(split_message) <= 1:
            continue
        if split_message[1] == "":
            battle.parse_message(split_message)
        elif split_message[1] in Player.MESSAGES_TO_IGNORE or split_message[1] in ("error", "bigerror"):
            continue
        elif split_message[1] == "request":
            if split_message[2]:
                battle.parse_request(json.loads(split_message[2]))
        elif split_message[1] == "win":
            battle.won_by(split_message[2])
        elif split_message[1] == "tie":
            battle.tied()
        else:
            battle.parse_message(split_message)
    return battle, request


def order_to_choice(order: BattleOrder, battle: Battle, request: dict[str, Any]) -> str:
    """Translate ``order`` into an index-based choice such as ``move 2 terastallize``."""

    if not isinstance(order, SingleBattleOrder):
        raise ValueError(f"Unsupported order {order!r}")
    if isinstance(order.order, Pokemon):
        for slot, member in enumerate(request["side"]["pokemon"], start=1):
            if battle.team.get(member["ident"]) is order.order:
                return f"switch {slot}"
        raise ValueError(f"{order.order.species} is not in the request's team")
    if not isinstance(order.order, Move):
        raise ValueError(f"Unsupported order {order!r}")
    moves = request["active"][0]["moves"]
    if order.order.id == "recharge" or len(moves) == 1:
        index = 1
    else:
        ids = [Move.retrieve_id(move["id"]) for move in moves]
        if order.order.id not in ids:
            raise ValueError(f"{order.order.id} is not in the request's moves {ids}")
        index = ids.index(order.order.id) + 1
    gimmicks = (
        (order.mega, "mega"),
        (order.z_move, "zmove"),
        (order.dynamax, "dynamax"),
        (order.terastallize, "terastallize"),
    )
    return " ".join([f"move {index}", *(word for enabled, word in gimmicks if enabled)])
