"""Tests for `Supervisor` space-pair gating (spec 006, sub-spec S02).

Before this fix, an unsupported (model, environment) pairing was accepted silently by
`configure()` (or, for `TDModel`, rejected by returning `False`, which `Supervisor`
ignored) and failed later with an unrelated error. `Supervisor` now checks
`model.supports_environment(environment)` before `configure()`, skips the mismatched
combination with a message naming both kinds, and continues with the rest.
"""

import logging
from pathlib import Path

import pytest

from hercule.config import EnvironmentConfig, HerculeConfig, ModelConfig
from hercule.supervisor import Supervisor


@pytest.mark.unit
def test_invalid_pairing_is_skipped_and_valid_pairing_still_runs(temp_test_dir: Path, caplog) -> None:
    """`simple_q_learning` requires (Discrete, Discrete); CartPole-v1 is (Box, Discrete).

    FrozenLake-v1 is (Discrete, Discrete), so it is a valid pairing for the same model
    and must still run to completion in the same invocation.
    """
    model_config = ModelConfig(name="simple_q_learning")
    invalid_env = EnvironmentConfig(name="CartPole-v1")
    valid_env = EnvironmentConfig(name="FrozenLake-v1")

    config = HerculeConfig(
        name="space_gating_test",
        environments=[invalid_env, valid_env],
        models=[model_config],
        learn_max_epoch=1,
        save_every_n_epoch=1,
        base_output_dir=temp_test_dir,
    )
    supervisor = Supervisor(config=config)

    with caplog.at_level(logging.WARNING):
        supervisor.execute_learn_phase()

    invalid_dir = config.get_directory_for(model_config, invalid_env)
    valid_dir = config.get_directory_for(model_config, valid_env)

    assert not (invalid_dir / "model.json").exists(), "the mismatched combination must not be trained"
    assert (valid_dir / "model.json").exists(), "the valid combination must still run"

    messages = " ".join(record.message for record in caplog.records)
    assert "simple_q_learning" in messages
    assert "CartPole-v1" in messages
    # The expected kinds (what the model supports) and the actual kinds (what the
    # environment offers) must both be named.
    assert "Discrete" in messages
    assert "Box" in messages


@pytest.mark.unit
def test_test_phase_also_skips_invalid_pairing(temp_test_dir: Path, caplog) -> None:
    """The same gate applies to `execute_test_phase`, not only to learning."""
    model_config = ModelConfig(name="simple_q_learning")
    invalid_env = EnvironmentConfig(name="CartPole-v1")

    config = HerculeConfig(
        name="space_gating_test_phase",
        environments=[invalid_env],
        models=[model_config],
        learn_max_epoch=1,
        test_epoch=1,
        save_every_n_epoch=1,
        base_output_dir=temp_test_dir,
    )
    supervisor = Supervisor(config=config)

    with caplog.at_level(logging.WARNING):
        supervisor.execute_test_phase()

    messages = " ".join(record.message for record in caplog.records)
    assert "simple_q_learning" in messages
    assert "CartPole-v1" in messages


@pytest.mark.unit
def test_sac_is_skipped_on_a_discrete_action_environment(temp_test_dir: Path, caplog) -> None:
    """The mirror case, from feature 007: SAC requires (Box, Box).

    `FrozenLake-v1` is (Discrete, Discrete) and `CartPole-v1` is (Box, Discrete), so
    both offer a discrete action space and both must be refused -- while the valid
    pairing in the same invocation still runs. This matters more for SAC than for the
    tabular models: a continuous policy handed a discrete action space would otherwise
    fail deep inside the action mapping, with an error naming neither the model nor
    the environment.
    """
    model_config = ModelConfig(name="sac")
    discrete_action_envs = [EnvironmentConfig(name="FrozenLake-v1"), EnvironmentConfig(name="CartPole-v1")]
    valid_env = EnvironmentConfig(name="AsymmetricOracle-v0")

    config = HerculeConfig(
        name="sac_space_gating",
        environments=[*discrete_action_envs, valid_env],
        models=[model_config],
        learn_max_epoch=1,
        save_every_n_epoch=1,
        base_output_dir=temp_test_dir,
    )
    supervisor = Supervisor(config=config)

    with caplog.at_level(logging.WARNING):
        supervisor.execute_learn_phase()

    for environment in discrete_action_envs:
        directory = config.get_directory_for(model_config, environment)
        assert not (directory / "model.json").exists(), f"{environment.name} must not be trained with sac"

    valid_dir = config.get_directory_for(model_config, valid_env)
    assert (valid_dir / "model.json").exists(), "the valid continuous pairing must still run"

    messages = " ".join(record.message for record in caplog.records)
    assert "sac" in messages
    assert "FrozenLake-v1" in messages and "CartPole-v1" in messages
    # Both sides named: what SAC accepts, and what the environment actually offers.
    assert "Box" in messages
    assert "Discrete" in messages
