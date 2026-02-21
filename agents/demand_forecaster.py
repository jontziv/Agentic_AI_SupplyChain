# agents/demand_forecaster.py - Fixed serialization version

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import subprocess
import sys
from typing import Dict, Any
from config import SupplyChainConfig
from state import SupplyChainState, make_serializable
from datetime import datetime
from state import make_serializable
from agents.tooling import tools_system_message, log_agent_action

class DemandForecasterAgent:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.agent_name = "Demand Forecaster"
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM with proper error handling"""
        try:
            if not self.config.GROQ_API_KEY:
                raise ValueError(f"{self.agent_name}: GROQ_API_KEY is required but not found")
            
            return ChatGroq(
                groq_api_key=self.config.GROQ_API_KEY,
                model_name=self.config.GROQ_MODEL,
                temperature=0.1,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            print(f"{self.agent_name}: Failed to initialize LLM: {e}")
            raise
    
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
    
    def generate_forecast_strategy(self, state: SupplyChainState) -> SupplyChainState:
        """Generate intelligent forecasting strategy based on data analysis"""
        
        print("Demand Forecaster: Starting forecast strategy generation...")
        
        ai_strategy = "Using default Prophet parameters"
        
        if self.llm:
            print("Demand Forecaster: Generating AI-powered forecasting strategy...")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert demand forecasting specialist.
                Based on the data analysis, create a comprehensive forecasting strategy that includes:
                1. Recommended forecasting approach and parameters
                2. Seasonal adjustment factors
                3. External factors to consider
                4. Confidence intervals and uncertainty quantification
                5. Model validation approach
                
                Consider the business context and provide actionable recommendations.
                Focus on practical parameters for Prophet model implementation."""),
                SystemMessage(content=tools_system_message(self.config)),
                HumanMessage(content=f"""
                Data Quality Issues: {state.get('data_quality_issues', [])}
                Seasonal Factors: {state.get('seasonal_factors', {})}
                Business Objectives: {state.get('business_objectives', {})}
                """)
            ])
            
            ai_response = self._safe_llm_call(prompt)
            if ai_response:
                ai_strategy = ai_response
                # Add message to state
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(AIMessage(content=f"{self.agent_name}: {ai_response}"))
        
        # Execute Prophet model with AI-recommended parameters
        print("Demand Forecaster: Running Prophet forecasting model...")
        log_agent_action(
            state, self.agent_name, "call_python_script",
            {"path": str(self.config.PROPHET_SCRIPT)}
        )
        forecast_results = self._execute_prophet_with_ai_params(ai_strategy)
        log_agent_action(
            state, self.agent_name, "call_python_script",
            {"path": str(self.config.PROPHET_SCRIPT)},
            status="completed",
            result_preview={"status": forecast_results.get("status"), "confidence": forecast_results.get("confidence_score")}
        )
        
        # CRITICAL: Make forecast results serializable
        state["demand_forecast"] = make_serializable(forecast_results)
        
        # Determine next action based on forecast quality
        confidence = forecast_results.get("confidence_score", 0)
        print(f"Demand Forecaster: Forecast confidence: {confidence}")
        
        if confidence < 0.6:
            print("Demand Forecaster: Low confidence - requiring human review")
            state["requires_human_review"] = True
            state["next_action"] = "human_review"
        else:
            state["next_action"] = "optimizing"
            
        state["current_step"] = "forecasting"
        
        return state
    
    def _execute_prophet_with_ai_params(self, ai_strategy: str) -> Dict[str, Any]:
        """Execute Prophet script with AI-recommended parameters"""
        try:
            print("Demand Forecaster: Executing Prophet.py script...")
            
            # Run Prophet script
            result = subprocess.run(
                [sys.executable, str(self.config.PROPHET_SCRIPT)],
                capture_output=True,
                text=True,
                cwd=self.config.ROOT_FOLDER,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("Demand Forecaster: Prophet execution successful")
                # Try to get confidence from Prophet output
                confidence = 0.85  # Default
                
                # Check if results file was created
                if self.config.DEMAND_FORECAST_RESULTS.exists():
                    try:
                        df = pd.read_csv(self.config.DEMAND_FORECAST_RESULTS)
                        # Calculate confidence based on MAE if available
                        if 'MAE' in df.columns and 'forecasted_demand' in df.columns:
                            avg_mae = float(df['MAE'].mean())
                            avg_demand = float(df['forecasted_demand'].mean())
                            if avg_demand > 0:
                                mae_ratio = avg_mae / avg_demand
                                confidence = float(max(0.5, 1.0 - mae_ratio))
                    except Exception as e:
                        print(f"Demand Forecaster: Could not calculate confidence from results: {e}")
                
                return {
                    "status": "success",
                    "output": result.stdout,
                    "confidence_score": confidence,
                    "ai_strategy": ai_strategy
                }
            else:
                print(f"Demand Forecaster: Prophet execution failed: {result.stderr}")
                return {
                    "status": "error",
                    "error": result.stderr,
                    "confidence_score": 0.0
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error", 
                "error": "Prophet script execution timed out",
                "confidence_score": 0.0
            }
        except Exception as e:
            print(f"Demand Forecaster: Execution error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "confidence_score": 0.0
            }