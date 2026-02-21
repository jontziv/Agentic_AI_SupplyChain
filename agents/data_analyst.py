# agents/data_analyst.py - Fixed serialization version

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from config import SupplyChainConfig
from state import SupplyChainState, make_serializable

from datetime import datetime
from state import make_serializable
from agents.tooling import tools_system_message, log_agent_action

class DataAnalystAgent:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.agent_name = "Data Analyst"
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM with proper error handling"""
        try:
            if not self.config.GROQ_API_KEY:
                print(f"{self.agent_name}: No GROQ API key found")
                return None
            
            return ChatGroq(
                groq_api_key=self.config.GROQ_API_KEY,
                model_name=self.config.GROQ_MODEL,
                temperature=0.1,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            print(f"{self.agent_name}: Failed to initialize LLM: {e}")
            return None
    
    def _safe_llm_call(self, prompt, **kwargs):
        """Make LLM call with proper error handling"""
        if not self.llm:
            return None
        
        try:
            messages = prompt.format_messages(**kwargs)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"{self.agent_name}: LLM call failed: {e}")
            return None
        
    def analyze_data_quality(self, state: SupplyChainState) -> SupplyChainState:
        """Analyze data quality and detect anomalies with robust fallbacks"""
        
        print("Data Analyst: Starting data quality analysis...")
        
        # Always perform actual data analysis first
        data_analysis = self._perform_data_analysis(state)        
        # Try AI analysis if LLM is available
        ai_insights = None
        if self.llm:
            print("Data Analyst: Running AI-powered analysis...")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert data analyst specializing in supply chain data.
                Analyze the provided data quality metrics and identify:
                1. Data completeness issues
                2. Outliers and anomalies  
                3. Seasonal patterns
                4. Data reliability concerns
                5. Recommendations for data preprocessing
                
                Provide your analysis in structured JSON format with these keys:
                - issues: list of data quality issues
                - seasonal_factors: dictionary of seasonal multipliers (Q1, Q2, Q3, Q4)
                - recommendations: list of actionable recommendations
                - confidence: float between 0-1"""),
                SystemMessage(content=tools_system_message(self.config)),
                HumanMessage(content=f"Data Analysis Results: {json.dumps(data_analysis, default=str)}")
            ])
            
            ai_response = self._safe_llm_call(prompt)
            if ai_response:
                ai_insights = self._extract_json_from_response(ai_response)
                # Add message to state
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(AIMessage(content=f"{self.agent_name}: {ai_response}"))
        else:
            print("Data Analyst: Using rule-based analysis only (no LLM available)")
        
        # Create comprehensive insights combining AI and rule-based analysis
        insights = self._create_comprehensive_insights(data_analysis, ai_insights)
        
        # CRITICAL: Make all data serializable before storing in state
        state["data_quality_issues"] = make_serializable(insights.get("issues", []))
        state["seasonal_factors"] = make_serializable(insights.get("seasonal_factors", {}))
        
        # Determine next action based on data quality
        critical_issues = [issue for issue in insights.get("issues", []) 
                         if any(keyword in issue.lower() for keyword in ['critical', 'missing', 'corrupt', 'invalid'])]
        
        if len(critical_issues) > 2:
            state["requires_human_review"] = True
            state["next_action"] = "human_review"
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"] = make_serializable(state["warnings"] + ["Critical data quality issues detected"])
        else:
            state["next_action"] = "forecasting"
            
        state["current_step"] = "analyzing"
        
        print(f"Data Analyst: Analysis complete. Found {len(insights.get('issues', []))} issues. Next: {state['next_action']}")
        
        return state
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract the FIRST JSON object from an LLM response, safely."""
        if not response:
            return {}
        try:
            import re, json

            # 1) Prefer a fenced ```json ... ``` block if present
            fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```",
                            response, re.DOTALL | re.IGNORECASE)
            if fence:
                return json.loads(fence.group(1))

            # 2) Otherwise, take the first non-greedy {...} block
            m = re.search(r"\{.*?\}", response, re.DOTALL)
            if m:
                return json.loads(m.group(0))

        except Exception as e:
            # keep same logging semantics you already use
            print(f"{self.agent_name}: Failed to extract JSON: {e}")

        # fall back to returning the raw text so downstream still works
        return {"analysis": response, "extracted": "partial"}
    
    def _perform_data_analysis(self, state: SupplyChainState = None) -> Dict[str, Any]:
        """Perform comprehensive data analysis with error handling"""
        analysis = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "files_analyzed": []
        }
        
        try:
            # Analyze demand forecast data
            if self.config.DEMAND_FORECAST_FILE.exists():
                df = pd.read_csv(self.config.DEMAND_FORECAST_FILE)
                preview = df.head(5).to_dict(orient="records")
                if state is not None:
                    log_agent_action(
                        state, self.agent_name, "read_csv_preview",
                        {"path": str(self.config.DEMAND_FORECAST_FILE), "n": 5},
                        result_preview=preview
                    )
                # Convert date column if it exists
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    try:
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    except:
                        pass
                
                analysis["demand_data"] = make_serializable({
                    "file_exists": True,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "date_range": self._get_date_range(df),
                    "numeric_summaries": self._get_numeric_summaries(df),
                    "preview": preview
                })
                if state is not None:
                    log_agent_action(
                        state, self.agent_name, "compute_stats",
                        {"path": str(self.config.DEMAND_FORECAST_FILE), "columns": list(df.columns)},
                        result_preview=analysis["demand_data"]["numeric_summaries"]
                    )
                analysis["files_analyzed"].append("demand_forecast_dataset.csv")
            else:
                analysis["demand_data"] = {"file_exists": False, "error": "File not found"}
            
            # Analyze inventory data
            if self.config.INVENTORY_OPTIMIZATION_FILE.exists():
                df = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_FILE)
                preview2 = df.head(5).to_dict(orient="records")
                if state is not None:
                    log_agent_action(
                        state, self.agent_name, "read_csv_preview",
                        {"path": str(self.config.INVENTORY_OPTIMIZATION_FILE), "n": 5},
                        result_preview=preview2
                    )
                analysis["inventory_data"] = make_serializable({
                    "file_exists": True,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_summaries": self._get_numeric_summaries(df),
                    "preview": preview2
                })
                if state is not None:
                    log_agent_action(
                        state, self.agent_name, "compute_stats",
                        {"path": str(self.config.INVENTORY_OPTIMIZATION_FILE), "columns": list(df.columns)},
                        result_preview=analysis["inventory_data"]["numeric_summaries"]
                    )
                analysis["files_analyzed"].append("inventory_optimisation_dataset.csv")
            else:
                analysis["inventory_data"] = {"file_exists": False, "error": "File not found"}
                
        except Exception as e:
            if "errors" not in analysis:
                analysis["errors"] = []
            analysis["errors"].append(f"Analysis error: {str(e)}")
            
        return make_serializable(analysis)
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract date range information safely"""
        try:
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                return {
                    "start": str(min_date),
                    "end": str(max_date),
                    "days": int((max_date - min_date).days)
                }
        except:
            pass
        return {"start": None, "end": None, "days": 0}
    
    def _get_numeric_summaries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summaries for numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summaries = {}
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    col_data = df[col]
                    summaries[col] = {
                        "mean": float(col_data.mean()) if pd.notna(col_data.mean()) else None,
                        "std": float(col_data.std()) if pd.notna(col_data.std()) else None,
                        "min": float(col_data.min()) if pd.notna(col_data.min()) else None,
                        "max": float(col_data.max()) if pd.notna(col_data.max()) else None,
                        "zeros": int((col_data == 0).sum()),
                        "negatives": int((col_data < 0).sum())
                    }
                return summaries
        except:
            pass
        return {}
    
    def _create_comprehensive_insights(self, data_analysis: Dict[str, Any], ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and AI insights"""
        
        issues = []
        seasonal_factors = {"Q1": 1.0, "Q2": 1.1, "Q3": 0.9, "Q4": 1.2}  # Default
        recommendations = []
        
        # Rule-based issue detection
        if not data_analysis.get("demand_data", {}).get("file_exists", False):
            issues.append("CRITICAL: Demand forecast dataset not found")
        
        if not data_analysis.get("inventory_data", {}).get("file_exists", False):
            issues.append("CRITICAL: Inventory optimization dataset not found")
        
        # Check for missing values
        demand_missing = data_analysis.get("demand_data", {}).get("missing_values", {})
        if any(count > 0 for count in demand_missing.values()):
            issues.append("Missing values detected in demand data")
        
        inventory_missing = data_analysis.get("inventory_data", {}).get("missing_values", {})
        if any(count > 0 for count in inventory_missing.values()):
            issues.append("Missing values detected in inventory data")
        
        # Check data freshness
        demand_data = data_analysis.get("demand_data", {})
        if demand_data.get("rows", 0) == 0:
            issues.append("No data rows found in demand dataset")
        elif demand_data.get("rows", 0) < 100:
            issues.append("Limited data in demand dataset - may affect forecast accuracy")
        
        # Incorporate AI insights if available
        if ai_insights:
            ai_issues = ai_insights.get("issues", [])
            ai_seasonal = ai_insights.get("seasonal_factors", {})
            ai_recs = ai_insights.get("recommendations", [])
            
            issues.extend(ai_issues)
            if ai_seasonal:
                seasonal_factors.update(ai_seasonal)
            recommendations.extend(ai_recs)
        
        result = {
            "issues": list(set(issues)),  # Remove duplicates
            "seasonal_factors": seasonal_factors,
            "recommendations": recommendations,
            "confidence": ai_insights.get("confidence", 0.8) if ai_insights else 0.7
        }
        
        return make_serializable(result)