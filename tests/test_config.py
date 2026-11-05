import pytest
from pydantic import ValidationError

from src.config import load_config, load_models


def test_pilot_loads():
    cfg = load_config("configs/pilot.yaml")
    assert cfg.dataset.n_questions == 20
    assert "multi_agent" in cfg.systems
    assert cfg.multi_agent.budget_split.workers == 0.70


def test_models_load():
    models = load_models("configs/models.yaml")
    assert "worker" in models and "judge" in models


def test_bad_budget_split_rejected():
    from src.config import BudgetSplit
    with pytest.raises(ValidationError):
        BudgetSplit(orchestrator=0.5, workers=0.5, synthesis=0.5)


def test_unknown_system_rejected():
    import yaml
    from src.config import ExperimentConfig
    raw = yaml.safe_load(open("configs/pilot.yaml"))
    raw["systems"] = ["not_a_system"]
    with pytest.raises(ValidationError):
        ExperimentConfig(**raw)

def test_long_context_pricing():
    m = load_models("configs/models.yaml")["worker"]
    short = m.cost_usd(50_000, 1_000)      # 0.005 + 0.0005
    long_ = m.cost_usd(150_000, 1_000)     # 0.075 + 0.0025
    assert abs(short - 0.0055) < 1e-9
    assert abs(long_ - 0.0775) < 1e-9