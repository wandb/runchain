# From weave code

import numpy as np
import random
import typing

value_fns: list[typing.Any] = [
    lambda steps: steps**2,
    lambda steps: np.cos(steps * 0.0001),
    lambda steps: np.sin(steps * 0.01),
    lambda steps: np.log(steps + 1),
    lambda steps: np.exp(steps * 0.0001),
    lambda steps: np.exp(-steps * 0.0001) * 1000,  # Simulate decreasing loss
    lambda steps: 1 - np.exp(-steps * 0.0001),  # Simulate increasing accuracy
    lambda steps: np.power(steps, -0.5)
    * 1000,  # Simulate decreasing loss with power-law decay
    lambda steps: np.tanh(steps * 0.0001),  # Simulate a metric converging to a value
    lambda steps: np.arctan(
        steps * 0.0001
    ),  # Simulate a metric converging to a value with a different curve
    lambda steps: np.piecewise(
        steps,
        [steps < len(steps) / 2, steps >= len(steps) / 2],
        [lambda steps: steps * 0.001, lambda steps: 1 - np.exp(-steps * 0.0001)],
    ),  # Simulate a two-stage training process
    lambda steps: np.sin(steps * 0.001)
    * np.exp(-steps * 0.0001),  # Sinusoidal oscillations with exponential decay
    lambda steps: (np.cos(steps * 0.001) + 1)
    * 0.5
    * (1 - np.exp(-steps * 0.0001)),  # Oscillations converging to increasing accuracy
    lambda steps: np.log(steps + 1)
    * (
        1 - np.exp(-steps * 0.0001)
    ),  # Logarithmic growth modulated by increasing accuracy
    lambda steps: np.random.random()
    * (
        1 - np.exp(-steps * 0.0001)
    ),  # Random constant value modulated by increasing accuracy
]


def random_metrics_template(n_metrics: int) -> list[int]:
    return [random.randrange(0, len(value_fns)) for _ in range(n_metrics)]


def random_metrics(
    n: int = 100000, n_metrics: int = 100, template: typing.Optional[list[int]] = None
) -> dict:
    # gpt-4 helped here.

    steps = np.arange(0, n, 1)
    data = {
        "step": steps,
        "string_col": np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), n),
    }

    for j in range(n_metrics):
        noise_fraction = random.random()
        if template is None:
            fn = random.choice(value_fns)
        else:
            fn = value_fns[template[j]]
        values = fn(steps)

        # Add different types of noise
        noise_type = random.choice(["uniform", "normal", "triangular"])
        if noise_type == "uniform":
            noise = np.random.uniform(low=-noise_fraction, high=noise_fraction, size=n)
        elif noise_type == "normal":
            noise = np.random.normal(scale=noise_fraction, size=n)
        elif noise_type == "triangular":
            noise = np.random.triangular(
                left=-noise_fraction, mode=0, right=noise_fraction, size=n
            )

        data[f"metric{j}"] = values + noise_fraction * values * noise

    return data
