from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PipelineConfig(BaseModel):
    tickers: List[str] = Field(default=["AAPL", "MSFT", "GOOGL", "SPY"])
    target: str = Field(default="AAPL_ret")
    start_date: str = Field(default="2015-01-01")
    end_date: str = Field(default="2025-12-31")
    max_lag: int = Field(default=5, ge=1, le=10)
    alpha: float = Field(default=0.05)
    indep_test: str = Field(default="rcot")
    n_regimes: int = Field(default=3, ge=2, le=5)
    horizon: int = Field(default=5, ge=1, le=20)
    context_length: int = Field(default=1024)
    initial_capital: float = Field(default=100000.0)
    causal_retrain_interval: int = Field(default=60)
    forecast_retrain_interval: int = Field(default=3)
    include_macro: bool = Field(default=True)
    include_factors: bool = Field(default=False)
    # Signal direction: "both" | "long_only" | "short_only"
    signal_direction: str = Field(default="both")


class StageStatus(BaseModel):
    status: str  # pending | running | completed | failed | skipped
    duration_sec: Optional[float] = None
    output_path: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None


class LogEntry(BaseModel):
    timestamp: str
    level: str
    stage: Optional[int] = None
    message: str


class PipelineRun(BaseModel):
    id: str
    status: str  # pending | running | completed | failed
    config: PipelineConfig
    current_stage: int = 0
    stage_statuses: Dict[int, StageStatus] = {}
    logs: List[LogEntry] = []
    created_at: str
    completed_at: Optional[str] = None
    results_available: bool = False


class RunListItem(BaseModel):
    id: str
    status: str
    target: str
    tickers: List[str]
    created_at: str
    completed_at: Optional[str] = None
    current_stage: int = 0


class WSEvent(BaseModel):
    run_id: str
    event: str
    stage: int
    stage_name: str
    progress: float
    message: str
    timestamp: str
    data: Optional[Any] = None


class GraphNode(BaseModel):
    id: str
    type: str  # target | feature | time_node
    nonstationary: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    directed: bool
    strength: float
    p_value: float
    validated: Optional[bool] = None


class GraphJSON(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
