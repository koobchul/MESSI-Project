import os
import pickle

import jax


CHECKPOINT_VERSION = 2


def save_checkpoint(
    path,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    carry,
    rollout_key,
    iteration,
    total_env_steps,
    episode,
    episode_steps,
    episode_return,
):
    """
    Save the complete MAPPO training state required for an exact
    mid-episode resume.

    Arrays are transferred to host memory before serialization so
    the checkpoint does not depend on a live accelerator device.
    """

    checkpoint = {
        "version": CHECKPOINT_VERSION,

        "actor_params": jax.device_get(
            actor_params
        ),

        "critic_params": jax.device_get(
            critic_params
        ),

        "actor_opt_state": jax.device_get(
            actor_opt_state
        ),

        "critic_opt_state": jax.device_get(
            critic_opt_state
        ),

        # --------------------------------------------------------
        # Environment / opponent state
        # --------------------------------------------------------
        "carry": jax.device_get(
            carry
        ),

        # Match-level learned-policy / environment randomness key.
        "rollout_key": jax.device_get(
            rollout_key
        ),

        # --------------------------------------------------------
        # Training counters
        # --------------------------------------------------------
        "iteration": int(
            iteration
        ),

        "total_env_steps": int(
            total_env_steps
        ),

        "episode": int(
            episode
        ),

        "episode_steps": int(
            episode_steps
        ),

        "episode_return": float(
            episode_return
        ),
    }

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temporary_path = (
        path + ".tmp"
    )

    # Atomic-style replacement:
    #
    # write new checkpoint completely first,
    # then replace the previous file.
    with open(
        temporary_path,
        "wb",
    ) as file:
        pickle.dump(
            checkpoint,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temporary_path,
        path,
    )


def load_checkpoint(
    path,
):
    """
    Load a complete MAPPO training checkpoint.
    """

    with open(
        path,
        "rb",
    ) as file:
        checkpoint = pickle.load(
            file
        )

    required_keys = {
        "version",

        "actor_params",
        "critic_params",

        "actor_opt_state",
        "critic_opt_state",

        "carry",
        "rollout_key",

        "iteration",
        "total_env_steps",

        "episode",
        "episode_steps",
        "episode_return",
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if (
        checkpoint["version"]
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{checkpoint['version']} "
            f"(expected {CHECKPOINT_VERSION})"
        )

    return checkpoint