from project.envs.messi_5v5 import make_5v5_env


def test_global_state_structure():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    # ============================================================
    # 2. Centralized global state
    # ============================================================
    global_state = env.global_state_view(
        reset.rollout
    )

    # ============================================================
    # 3. MESSI typed flatten
    # ============================================================
    flat_state, state_spec = env.flatten_global_state(
        global_state
    )

    print("\n=== Global State Flat Blocks ===")

    print(
        "float32 shape:",
        flat_state.float32.shape,
    )

    print(
        "int32 shape:",
        flat_state.int32.shape,
    )

    print(
        "boolean shape:",
        flat_state.boolean.shape,
    )

    print("\n=== Global State Spec ===")

    print(
        "float32 size:",
        state_spec.float32_size,
    )

    print(
        "int32 size:",
        state_spec.int32_size,
    )

    print(
        "boolean size:",
        state_spec.boolean_size,
    )

    # ============================================================
    # 4. Basic checks
    # ============================================================
    assert flat_state.float32.ndim == 1
    assert flat_state.int32.ndim == 1
    assert flat_state.boolean.ndim == 1

    assert (
        flat_state.float32.shape[0]
        == state_spec.float32_size
    )

    assert (
        flat_state.int32.shape[0]
        == state_spec.int32_size
    )

    assert (
        flat_state.boolean.shape[0]
        == state_spec.boolean_size
    )

    # ============================================================
    # 5. Inspect Global State leaf layout
    # ============================================================
    print("\n=== Global State Leaf Layout ===")

    for leaf in state_spec.leaves:
        print(
            f"{leaf.path:55s} "
            f"block={leaf.block:8s} "
            f"offset={leaf.offset:3d} "
            f"size={leaf.size:3d} "
            f"shape={leaf.trailing_shape}"
        )