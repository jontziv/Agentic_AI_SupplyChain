import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

load_dotenv()

class WorkflowStatus(Enum):
    IDLE = "idle"
    DATA_ANALYSIS = "data_analysis"
    DEMAND_FORECASTING = "demand_forecasting"
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    ERROR = "error"

@dataclass
class SupplyChainConfig:
    # File paths
    ROOT_FOLDER: Path = Path(".")
    DATA_FOLDER: Path = Path("data")
    
    # Input files
    DEMAND_FORECAST_FILE: Path = ROOT_FOLDER / "demand_forecast_dataset.csv"
    INVENTORY_OPTIMIZATION_FILE: Path = ROOT_FOLDER / "inventory_optimisation_dataset.csv"
    
    # Model scripts
    PROPHET_SCRIPT: Path = ROOT_FOLDER / "Prophet.py"
    REORDERPOINT_SCRIPT: Path = ROOT_FOLDER / "Reorderpoint.py"
    
    # Output files
    DEMAND_FORECAST_RESULTS: Path = DATA_FOLDER / "demand_forecast_results.csv"
    INVENTORY_OPTIMIZATION_RESULTS: Path = DATA_FOLDER / "inventory_optimization_results.csv"
    
    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    # Business rules
    MAX_SAFETY_STOCK_INCREASE: float = 0.5  # 50% max increase
    MIN_SAFETY_STOCK_DAYS: int = 7
    MAX_SAFETY_STOCK_DAYS: int = 90
    SERVICE_LEVEL_TARGET: float = 0.95
    
    # Risk thresholds
    HIGH_RISK_FORECAST_DEVIATION: float = 0.3  # 30% deviation
    HIGH_RISK_INVENTORY_TURNOVER: float = 2.0
    CRITICAL_ITEM_THRESHOLD: float = 0.8  # 80% of revenue
    
    def __post_init__(self):
            self.DATA_FOLDER.mkdir(exist_ok=True)
    
    def validate_config(self):
        issues = []
        # Required input scripts
        for script in [self.PROPHET_SCRIPT, self.REORDERPOINT_SCRIPT]:
            if not Path(script).exists():
                issues.append(f"Missing required script: {script}")
        # Required CSVs
        if not self.DEMAND_FORECAST_FILE.exists():
            issues.append(f"Missing dataset: {self.DEMAND_FORECAST_FILE}")
        if not self.INVENTORY_OPTIMIZATION_FILE.exists():
            issues.append(f"Missing dataset: {self.INVENTORY_OPTIMIZATION_FILE}")
        # Optional (warn) – GROQ
        if not self.GROQ_API_KEY:
            issues.append("GROQ_API_KEY not set (AI analysis will use fallbacks)")
        return issues

    
    def get_file_info(self):
        info = {}
        try:
            if self.DEMAND_FORECAST_FILE.exists():
                df = pd.read_csv(self.DEMAND_FORECAST_FILE, nrows=5)
                rows = sum(1 for _ in open(self.DEMAND_FORECAST_FILE)) - 1
                info["demand_data"] = {"rows": max(rows, 0), "preview": df.to_dict("records")}
            else:
                info["demand_data"] = {"error": "not_found"}
        except Exception as e:
            info["demand_data"] = {"error": str(e)}

        try:
            if self.INVENTORY_OPTIMIZATION_FILE.exists():
                df = pd.read_csv(self.INVENTORY_OPTIMIZATION_FILE, nrows=5)
                rows = sum(1 for _ in open(self.INVENTORY_OPTIMIZATION_FILE)) - 1
                info["inventory_data"] = {"rows": max(rows, 0), "preview": df.to_dict("records")}
            else:
                info["inventory_data"] = {"error": "not_found"}
        except Exception as e:
            info["inventory_data"] = {"error": str(e)}

        return info