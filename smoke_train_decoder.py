"""Train only the Fly CNS action decoder on one fixed synthetic example."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from flycns.action_decoder import ACTION_COUNT
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


@dataclass(frozen=True)
class LearningSmokeResult:
    target_action: int
    before_score: float
    before_probability: float
    after_score: float
    after_probability: float
    changed_parameters: tuple[str, ...]


def run_smoke(
    *, seed: int = 7, target_action: int = 7, steps: int = 3
) -> LearningSmokeResult:
    """Fit one synthetic input and report how its target action changes."""

    if not 0 <= target_action < ACTION_COUNT:
        raise ValueError(f"target_action must be in [0, {ACTION_COUNT})")
    if steps <= 0:
        raise ValueError("steps must be positive")

    # A fixed, nonnegative activity pattern; no graph or sensory simulation runs.
    activity = torch.arange(DESCENDING_ACTIVITY_SIZE, dtype=torch.float32).remainder(17) / 16
    original_activity = activity.clone()
    target = torch.tensor([target_action], dtype=torch.long)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder = TrainableActionDecoder()
        original_parameters = {
            name: parameter.detach().clone()
            for name, parameter in decoder.named_parameters()
        }

        def target_values() -> tuple[float, float]:
            with torch.no_grad():
                logits = decoder(activity)
                return (
                    float(logits[target_action]),
                    float(logits.softmax(dim=-1)[target_action]),
                )

        before_score, before_probability = target_values()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.005)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = decoder(activity)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
        after_score, after_probability = target_values()

        changed_parameters = tuple(
            name
            for name, parameter in decoder.named_parameters()
            if not torch.equal(original_parameters[name], parameter.detach())
        )

    if not torch.equal(activity, original_activity):
        raise AssertionError("Synthetic activity changed during training")
    if set(changed_parameters) != set(original_parameters):
        raise AssertionError("Expected only the decoder's weight and bias to change")
    if after_probability <= before_probability:
        raise AssertionError("Target action did not become more probable")

    return LearningSmokeResult(
        target_action=target_action,
        before_score=before_score,
        before_probability=before_probability,
        after_score=after_score,
        after_probability=after_probability,
        changed_parameters=changed_parameters,
    )


if __name__ == "__main__":
    result = run_smoke()
    print(f"Target action: {result.target_action}")
    print(
        f"Before: score={result.before_score:.6f}, "
        f"probability={result.before_probability:.6f}"
    )
    print(
        f"After:  score={result.after_score:.6f}, "
        f"probability={result.after_probability:.6f}"
    )
    print(f"Changed decoder parameters: {', '.join(result.changed_parameters)}")
