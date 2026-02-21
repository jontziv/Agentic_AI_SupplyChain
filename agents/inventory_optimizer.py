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

class InventoryOptimizerAgent:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.agent_name = "Inventory Optimizer"
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
    
    def optimize_inventory_parameters(self, state: SupplyChainState) -> SupplyChainState:
        """Optimize inventory parameters using AI-driven approach"""
        
        print(f"{self.agent_name}: Starting inventory optimization...")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert inventory optimization specialist.
            Based on the demand forecast results, optimize inventory parameters:
            1. Reorder points for each SKU
            2. Safety stock levels considering service level targets
            3. MOQ optimization balancing holding costs and ordering costs
            4. Lead time variability adjustments
            5. ABC analysis for differentiated strategies
            
            Consider business constraints and provide justification for recommendations."""),
            SystemMessage(content=tools_system_message(self.config)),
            HumanMessage(content=f"""
            Demand Forecast: {state.get('demand_forecast', {})}
            Service Level Target: {self.config.SERVICE_LEVEL_TARGET}
            Business Constraints: {state.get('supplier_constraints', {})}
            Seasonal Factors: {state.get('seasonal_factors', {})}
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            
            # Add message to state
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                AIMessage(content=f"{self.agent_name}: {response.content}")
            )
            
            # Execute optimization with AI recommendations
            log_agent_action(
                state, self.agent_name, "call_python_script",
                {"path": str(self.config.REORDERPOINT_SCRIPT)}
            )
            optimization_results = self._execute_optimization_with_ai_params(response.content)
            log_agent_action(
                state, self.agent_name, "call_python_script",
                {"path": str(self.config.REORDERPOINT_SCRIPT)},
                status="completed",
                result_preview={"status": optimization_results.get("status"), "score": optimization_results.get("optimization_score")}
            )
            state["inventory_optimization"] = make_serializable(optimization_results)
            
            state["next_action"] = "risk_assessment"
            state["current_step"] = "optimizing"
            
            print(f"{self.agent_name}: Optimization completed successfully")
            
        except Exception as e:
            print(f"{self.agent_name}: Error during optimization: {e}")
            state["inventory_optimization"] = make_serializable({
                "status": "error",
                "error": str(e),
                "optimization_score": 0.0
            })
            state["requires_human_review"] = True
            state["next_action"] = "human_review"
        
        return state
    
    def _execute_optimization_with_ai_params(self, ai_recommendations: str) -> Dict[str, Any]:
        """Execute reorder point optimization with AI recommendations"""
        try:
            print(f"{self.agent_name}: Executing Reorderpoint.py script...")
            
            result = subprocess.run(
                [sys.executable, str(self.config.REORDERPOINT_SCRIPT)],
                capture_output=True,
                text=True,
                cwd=self.config.ROOT_FOLDER,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"{self.agent_name}: Reorderpoint script execution successful")
                return {
                    "status": "success",
                    "output": result.stdout,
                    "ai_recommendations": ai_recommendations,
                    "optimization_score": 0.9
                }
            else:
                print(f"{self.agent_name}: Reorderpoint script execution failed: {result.stderr}")
                return {
                    "status": "error",
                    "error": result.stderr,
                    "optimization_score": 0.0
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Reorderpoint script execution timed out",
                "optimization_score": 0.0
            }
        except Exception as e:
            print(f"{self.agent_name}: Execution error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "optimization_score": 0.0
            }