from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import pandas as pd
import numpy as np
from datetime import datetime
import json

def make_serializable(obj):
    """Convert numpy types and other non-serializable objects to Python native types"""
    if obj is None:
        return None
    elif isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return make_serializable(obj.item())
    elif hasattr(obj, 'tolist'):  # Handle other array-like objects
        return make_serializable(obj.tolist())
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return make_serializable(obj.__dict__)
    else:
        # For any other type, try to convert to string as last resort
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)

class SupplyChainState(TypedDict):
    # Message history for agent communication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Data processing
    raw_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    data_quality_issues: List[str]
    
    # Analysis results
    demand_forecast: Optional[Dict[str, Any]]
    inventory_optimization: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[Dict[str, Any]]
    confidence_scores: Dict[str, Any]
    
    # Human interaction
    human_feedback: Optional[Dict[str, Any]]
    approval_status: Optional[str]
    rejection_reasons: List[str]
    
    # Workflow control
    current_step: str
    next_action: Optional[str]
    requires_human_review: bool
    
    # Business context
    seasonal_factors: Dict[str, Any]
    supplier_constraints: Dict[str, Any]
    business_objectives: Dict[str, Any]
    
    # Audit trail
    decision_history: List[Dict[str, Any]]
    agent_actions: List[Dict[str, Any]]
    
    # Error handling
    errors: List[str]
    warnings: List[str]