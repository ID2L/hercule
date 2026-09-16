"""User Story 3 (P2): inspect, replay and compare a continuous agent.

Phase 5 of `specs/007-sac-continuous-control/tasks.md` (T055-T057). Story 1 and its
tests certify that SAC trains correctly; this file certifies that the rest of the
framework -- `hercule play`, `hercule report`, grid expansion and resume -- treats a
SAC run as an ORDINARY run, exactly as it does for every discrete-action model. None
of these three tests touches SAC's learning algorithm itself (that is Phase 4's job);
each is here because SAC is the framework's first model with more than one network,
more than one optimizer and a persisted scalar (the temperature) adapted outside any
network's own parameters, and the shared machinery has never been exercised against
that shape before.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import hercule.environnements  # noqa: F401  -- registers the oracle
from hercule.config import EnvironmentConfig, HerculeConfig, HyperParameter, ModelConfig
from hercule.environnements.oracle import ENVIRONMENT_ID, EPISODE_LENGTH
from hercule.models import model_file_name
from hercule.models.sac import SACModel
from hercule.reports import generate_report
from hercule.run import run_info_file_name
from hercule.supervisor import Supervisor, environment_file_name


# ---------------------------------------------------------------------------- T055


@pytest.mark.integration
def test_hercule_play_replays_a_trained_sac_agent_within_bounds(tmp_path: Path) -> None:
    """`hercule play`'s own code path replays a trained SAC agent within bounds (FR-010).

    `controller.play_interactive()` never sees the training config: it builds a
    FRESH model, calls `model.configure(env, {})` -- defaults only -- then
    `model.load_from_dict()` to hydrate the trained weights, then drives the episode
    with `model.begin_episode()` followed by `model.predict()` per step. This test
    reproduces exactly that sequence without invoking the `hercule play` CLI command
    itself, which loops until interrupted by design and would never return.

    Why this needs its own test rather than trusting the unit suite: every check in
    `test_sac_structure.py` and `test_action_mapping.py` calls `act()`/`predict()` on
    a model that was itself just configured and trained -- never on a model rebuilt
    from a save/load round trip through a DIFFERENT configuration, which is what
    `hercule play` actually does. A model whose `act()`/`predict()` returned the
    policy's own NORMALISED `[-1, 1]^d` coordinates instead of mapping them through
    `to_env_action()` would submit exactly those normalised values to the
    environment; on this oracle's asymmetric bounds (`[-1, 1] x [0, 4]`) that maps
    dimension 1 into `[-1, 1]` against a declared range of `[0, 4]`, out of bounds for
    every negative sample, while `act()`/`predict()` never left the model in
    isolation, so every existing unit test would still pass unchanged.
    """
    train_env = gym.make(ENVIRONMENT_ID)
    trained = SACModel()
    assert trained.configure(
        train_env,
        {"seed": 7, "learning_starts": 0, "batch_size": 8, "replay_buffer_size": 256, "step_modulo": 1},
    )
    for _ in range(3):
        trained.run_epoch(train_mode=True)

    directory = tmp_path / "sac_trained"
    trained.save(directory)

    with open(directory / model_file_name, encoding="utf-8") as f:
        model_data = json.load(f)

    # A FRESH model, configured with DEFAULT hyperparameters -- `hercule play` has no
    # access to the YAML the run was trained under, only the saved model file.
    play_env = gym.make(ENVIRONMENT_ID)
    played = SACModel()
    assert played.configure(play_env, {})
    played.load_from_dict(model_data)

    space = play_env.action_space
    observation, _ = play_env.reset(seed=0)
    played.begin_episode()

    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = played.predict(observation)
        action_array = np.asarray(action, dtype=np.float32)
        assert action_array.shape == space.shape, f"action shape {action_array.shape} != space shape {space.shape}"
        assert space.contains(action_array), f"{action_array} is outside the declared bounds {space}"
        observation, _, terminated, truncated, _ = play_env.step(action)
        steps += 1
        assert steps <= EPISODE_LENGTH, "the episode did not reach its natural end within the oracle's own horizon"

    # The oracle always truncates on a time limit and never terminates (its own
    # contract, `environnements/oracle.py`); asserting this confirms the loop above
    # actually ran a full, naturally-ending episode rather than exiting early.
    assert truncated
    assert not terminated


# ---------------------------------------------------------------------------- T056


@pytest.mark.integration
def test_report_generation_consumes_sac_runs_with_no_change_to_reports(temp_test_dir: Path) -> None:
    """FR-030: existing report generation consumes a grid of SAC runs, unmodified.

    SAC's `model.json`/`run_info.json`/`environment.json` are written by the exact
    same shared code every other model uses (`RLModel.save`, `Runner.save`,
    `save_environment`), and `build_run_table()` never opens `model.json` at all
    (CLAUDE.md: "never `model.json`"), so nothing about SAC's own file contents --
    five networks, four optimizers, a persisted temperature -- should matter to
    report generation. A failure here means either that shared writing path was
    broken by this feature, or a SAC run directory does not satisfy
    `is_valid_experiment_directory`/`build_run_table`'s expectations the way every
    other model's already does -- in which case Story 3's second scenario
    ("`hercule report` ... generates a comparative report") is unmet, and it would be
    unmet specifically because of something this feature introduced, since the test
    changes nothing under `src/hercule/reports/`.
    """
    model_config = ModelConfig(
        name="sac",
        hyperparameters=[
            HyperParameter(key="tau", value=[0.01, 0.05]),
            HyperParameter(key="learning_starts", value=0),
            HyperParameter(key="batch_size", value=8),
            HyperParameter(key="replay_buffer_size", value=256),
            HyperParameter(key="seed", value=9),
        ],
    )
    raw_config = HerculeConfig(
        name="sac_report_test",
        environments=[EnvironmentConfig(name=ENVIRONMENT_ID)],
        models=[model_config],
        learn_max_epoch=2,
        test_epoch=2,
        save_every_n_epoch=2,
        base_output_dir=temp_test_dir,
    )
    config = raw_config.expand_variants()
    assert len(config.models) == 2, "the grid must actually produce two SAC runs to compare"

    supervisor = Supervisor(config=config)
    supervisor.execute_learn_phase()
    supervisor.execute_test_phase()

    env_config = config.get_environment_configs()[0]
    group_dir = config.base_output_dir / config.name / ENVIRONMENT_ID / env_config.get_hyperparameters_signature()
    assert group_dir.is_dir()

    bundle = generate_report(group_dir, execute=False, render_pdf=False)

    assert bundle.skipped_groups == [], f"the SAC group was skipped: {bundle.skipped_groups}"
    assert bundle.report_count == 1
    artifact = bundle.reports[0]
    assert artifact.runs_loaded == 2, "both SAC runs must load; a lower count means one was silently dropped"
    assert artifact.source.exists()
    assert artifact.notebook.exists()

    content = artifact.source.read_text(encoding="utf-8")
    assert "sac" in content, "the report never names the model family it is supposed to be ranking"


# ---------------------------------------------------------------------------- T057


@pytest.mark.integration
def test_a_hyperparameter_list_expands_sac_into_independent_run_directories(temp_test_dir: Path) -> None:
    """FR-021, User Story 1 scenario 1: SAC participates in ordinary grid expansion.

    A failure here means SAC's hyperparameters (`tau` in particular, new to this
    feature) do not flow through `ModelConfig.expand_variants()` and
    `get_hyperparameters_signature()` the same way every other model's do -- e.g. a
    hyperparameter whose type or bounds interact badly with signature generation, or
    a config that silently trains only one variant.
    """
    model_config = ModelConfig(
        name="sac",
        hyperparameters=[
            HyperParameter(key="tau", value=[0.01, 0.05]),
            HyperParameter(key="learning_starts", value=0),
            HyperParameter(key="batch_size", value=8),
            HyperParameter(key="replay_buffer_size", value=256),
            HyperParameter(key="seed", value=11),
        ],
    )
    raw_config = HerculeConfig(
        name="sac_grid_test",
        environments=[EnvironmentConfig(name=ENVIRONMENT_ID)],
        models=[model_config],
        learn_max_epoch=2,
        save_every_n_epoch=2,
        base_output_dir=temp_test_dir,
    )
    config = raw_config.expand_variants()
    expanded_models = config.models
    assert len(expanded_models) == 2
    assert len({m.get_hyperparameters_signature() for m in expanded_models}) == 2, (
        "the two variants must land under DISTINCT signatures"
    )

    Supervisor(config=config).execute_learn_phase()

    env_config = config.get_environment_configs()[0]
    directories = [config.get_directory_for(model, env_config) for model in expanded_models]
    assert len({str(d) for d in directories}) == 2, "the two variants must land in DISTINCT directories"
    for directory in directories:
        assert (directory / environment_file_name).exists()
        assert (directory / model_file_name).exists()
        assert (directory / run_info_file_name).exists()


@pytest.mark.integration
def test_resuming_at_a_higher_epoch_ceiling_continues_rather_than_restarts(temp_test_dir: Path) -> None:
    """FR-021/FR-026 (User Story 1 scenario 2): resume continues, including the temperature.

    Two things must both be true of a re-run at a higher `learn_max_epoch`, and
    either could fail independently:

    1. Training continues from the recorded epoch (`Runner.learn()`'s own
       `range(learning_ongoing_epoch, max_epoch)` contract) rather than restarting
       every episode from epoch 0 -- checked here by asserting the FIRST phase's own
       recorded episodes are byte-for-byte untouched by the second phase, not merely
       that the final epoch count is the expected number, which a full restart would
       also produce.
    2. A quantity that was being ADAPTED during training -- SAC's learned
       temperature, `model._log_alpha` -- resumes from its stored value rather than
       reverting to the value `init_temperature` would give it on a fresh
       `configure()`. This is specific to an off-policy actor-critic model: unlike a
       discrete model's weights, which are only ever set by `load_state_dict()`, the
       temperature is a plain tensor that `_build_networks()` re-creates at its
       INITIAL value on every `configure()` call -- `Supervisor.execute_learn_phase()`
       always calls `configure()` before `load()`, so a `_load_extra_state()` that
       failed to restore it would leave the model training with a fresh temperature
       every single re-run, silently discarding the whole point of automatic
       adjustment across a resume.
    """
    model_config = ModelConfig(
        name="sac",
        hyperparameters=[
            HyperParameter(key="learning_starts", value=0),
            HyperParameter(key="batch_size", value=8),
            HyperParameter(key="replay_buffer_size", value=256),
            HyperParameter(key="seed", value=5),
        ],
    )
    env_config = EnvironmentConfig(name=ENVIRONMENT_ID)
    common_kwargs = {
        "name": "sac_resume_test",
        "environments": [env_config],
        "models": [model_config],
        "base_output_dir": temp_test_dir,
        # Large enough that no mid-training checkpoint fires; only the guaranteed
        # save at the end of Runner.learn() writes anything, keeping the two phases'
        # writes easy to reason about independently.
        "save_every_n_epoch": 100,
    }

    first_config = HerculeConfig(learn_max_epoch=2, **common_kwargs)
    Supervisor(config=first_config).execute_learn_phase()

    directory = first_config.get_directory_for(model_config, env_config)

    with open(directory / run_info_file_name, encoding="utf-8") as f:
        run_info_after_first = json.load(f)
    assert run_info_after_first["learning_ongoing_epoch"] == 2
    assert len(run_info_after_first["learning_metrics"]) == 2

    with open(directory / model_file_name, encoding="utf-8") as f:
        model_data_after_first = json.load(f)
    log_alpha_after_first = model_data_after_first["log_alpha"]

    default_hyperparams = SACModel().get_default_hyperparameters_typed()
    initial_log_alpha = float(np.log(default_hyperparams.init_temperature))
    assert log_alpha_after_first != pytest.approx(initial_log_alpha), (
        "the temperature never moved during the first phase, so this test cannot tell a "
        "correct resume from a broken one -- widen the warmup/batch settings above"
    )

    # Reproduce exactly what Supervisor.execute_learn_phase() does for the SECOND
    # phase, up to (not including) taking any further gradient step: create a FRESH
    # model, `configure()` it (which re-initialises `_log_alpha` to `log(init_temperature)`
    # again), then `load()` the stored checkpoint. This isolates whether `load()`
    # alone restores the temperature, independently of anything a further training
    # step might do to it.
    reload_env = gym.make(ENVIRONMENT_ID)
    reloaded = SACModel()
    assert reloaded.configure(reload_env, model_config.get_hyperparameters_dict())
    assert float(reloaded._log_alpha.item()) == pytest.approx(initial_log_alpha), (
        "sanity check: configure() alone must give the freshly-initialised temperature"
    )
    reloaded.load(directory)
    assert float(reloaded._log_alpha.item()) == pytest.approx(log_alpha_after_first), (
        "configure() followed by load() -- the exact sequence Supervisor uses -- must restore "
        "the STORED temperature, not leave it at the freshly configured initial value"
    )

    second_config = HerculeConfig(learn_max_epoch=5, **common_kwargs)
    Supervisor(config=second_config).execute_learn_phase()

    with open(directory / run_info_file_name, encoding="utf-8") as f:
        run_info_after_second = json.load(f)
    assert run_info_after_second["learning_ongoing_epoch"] == 5
    assert len(run_info_after_second["learning_metrics"]) == 5
    # A restart-from-zero disguised as a resume would also end at epoch 5 with 5
    # recorded episodes -- what it CANNOT do is leave the first two episodes' own
    # recorded results untouched, since a restart recomputes them from a fresh model.
    assert run_info_after_second["learning_metrics"][:2] == run_info_after_first["learning_metrics"]

    with open(directory / model_file_name, encoding="utf-8") as f:
        model_data_after_second = json.load(f)
    log_alpha_after_second = model_data_after_second["log_alpha"]
    assert log_alpha_after_second != pytest.approx(initial_log_alpha), (
        "after resuming and training further, the stored temperature reverted to the initial "
        "value instead of continuing adaptation from where the first phase left it"
    )
