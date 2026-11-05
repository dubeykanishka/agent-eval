from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

SystemName = Literal[
    "closed_book", "rag", "single_agent", "single_agent_compressed", "multi_agent"
]


class DatasetCfg(BaseModel):
    name: str
    split: Literal["dev", "test"]
    n_questions: int = Field(gt=0)
    pool_sizes: list[int]


class RagCfg(BaseModel):
    chunk_tokens: int = Field(default=300, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    top_k: int = Field(default=8, gt=0)


class BudgetSplit(BaseModel):
    orchestrator: float
    workers: float
    synthesis: float

    @model_validator(mode="after")
    def must_sum_to_one(self):
        total = self.orchestrator + self.workers + self.synthesis
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"budget_split must sum to 1.0, got {total}")
        return self


class MultiAgentCfg(BaseModel):
    n_workers: int = Field(default=4, gt=0)
    budget_split: BudgetSplit


class ExperimentConfig(BaseModel):
    experiment_name: str
    seed: int
    worker_model: str
    judge_model: str
    dataset: DatasetCfg
    systems: list[SystemName]
    budgets_tokens: list[int]
    max_steps: int = Field(gt=0)
    reserve_tokens: int = Field(ge=0)
    concurrency: int = Field(gt=0)
    spend_limit_usd: float = Field(gt=0)
    rag: RagCfg = RagCfg()
    multi_agent: MultiAgentCfg

    @model_validator(mode="after")
    def budgets_are_sane(self):
        smallest = min(self.budgets_tokens)
        if self.reserve_tokens >= smallest:
            raise ValueError("reserve_tokens must be smaller than the smallest budget")
        return self


class ModelSpec(BaseModel):
    model_id: str
    usd_per_million_in: float = Field(ge=0)
    usd_per_million_out: float = Field(ge=0)
    long_context_threshold: int | None = None
    usd_per_million_in_long: float | None = None
    usd_per_million_out_long: float | None = None
    checked_on: str

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """Cost of ONE API call. The whole call moves to the long-context
        rate if its prompt exceeds the threshold."""
        long_call = (
            self.long_context_threshold is not None
            and tokens_in > self.long_context_threshold
        )
        rate_in = self.usd_per_million_in_long if long_call else self.usd_per_million_in
        rate_out = self.usd_per_million_out_long if long_call else self.usd_per_million_out
        return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return ExperimentConfig(**yaml.safe_load(f))


def load_models(path: str | Path = "configs/models.yaml") -> dict[str, ModelSpec]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    models = {name: ModelSpec(**spec) for name, spec in raw.items()}
    for name, m in models.items():
        if m.usd_per_million_in == 0 and m.usd_per_million_out == 0:
            print(f"WARNING: prices for '{name}' are still 0.0 in models.yaml")
    return models