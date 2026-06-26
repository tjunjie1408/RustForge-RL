from benchmarks.sb3_comparison import config


def test_matched_config_values():
    assert config.STEP_BUDGET == 50_000
    assert config.SEEDS == list(range(10))
    assert config.BATCH_SIZE == 32
    assert config.BUFFER_SIZE == 10_000
    assert config.LEARNING_STARTS == 128
    assert config.GAMMA == 0.99
    assert config.LR == 1e-3
    assert config.HIDDEN_DIM == 64
    assert config.TARGET_UPDATE == 100
    assert config.EPS_INITIAL == 1.0 and config.EPS_FINAL == 0.05
    assert config.EPS_DECAY_STEPS == 2_000
    assert config.SOLVED_THRESHOLD == 475.0
    assert config.SOLVED_WINDOW == 100


def test_sb3_kwargs_match_rustforge():
    kw = config.sb3_kwargs(seed=7)
    assert kw["batch_size"] == 32
    assert kw["buffer_size"] == 10_000
    assert kw["learning_starts"] == 128
    assert kw["train_freq"] == 1
    assert kw["gradient_steps"] == 1
    assert kw["target_update_interval"] == 100
    assert kw["policy_kwargs"] == {"net_arch": [64]}
    assert kw["device"] == "cpu"
    assert kw["seed"] == 7
    # ε linear schedule maps exactly onto SB3's exploration_fraction
    assert kw["exploration_initial_eps"] == 1.0
    assert kw["exploration_final_eps"] == 0.05
    assert abs(kw["exploration_fraction"] - 2_000 / 50_000) < 1e-12
