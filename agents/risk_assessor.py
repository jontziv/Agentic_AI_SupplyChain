from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json
from typing import Dict, Any
from config import SupplyChainConfig
from state import SupplyChainState, make_serializable

from datetime import datetime
from state import make_serializable
from agents.tooling import tools_system_message, log_agent_action



class RiskAssessorAgent:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.agent_name = "Risk Assessor"
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
                temperature=0.2,
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
    
    def assess_supply_chain_risks(self, state: SupplyChainState) -> SupplyChainState:
        """Comprehensive risk assessment of proposed recommendations"""
        
        print(f"{self.agent_name}: Starting risk assessment...")
        
        # Perform quantitative risk analysis first
        risk_analysis = self._perform_quantitative_risk_analysis(state)
        tools_system_message(state, self.agent_name, "compute_stats", {"source": "risk_metrics"}, result_preview=risk_analysis)
        
        # Try AI analysis if LLM is available
        if self.llm:
            print(f"{self.agent_name}: Running AI-powered risk analysis...")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert supply chain risk analyst.
                Perform a comprehensive risk assessment of the proposed inventory optimization:
                1. Stockout risk analysis by SKU
                2. Overstock and obsolescence risk
                3. Supplier reliability and lead time risks
                4. Financial impact analysis
                5. Scenario analysis (best/worst/most likely cases)
                6. Risk mitigation recommendations
                
                Classify risks as LOW/MEDIUM/HIGH and provide specific mitigation strategies.
                Format your response as structured analysis with clear risk classifications."""),
                SystemMessage(content=tools_system_message(self.config)),
                HumanMessage(content=f"""
                Demand Forecast Results: {state.get('demand_forecast', {})}
                Inventory Optimization: {state.get('inventory_optimization', {})}
                Business Objectives: {state.get('business_objectives', {})}
                Quantitative Risk Metrics: {risk_analysis}
                """)
            ])
            
            ai_response = self._safe_llm_call(prompt)
            if ai_response:
                # Add message to state
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(
                    AIMessage(content=f"{self.agent_name}: {ai_response}")
                )
                
                # Enhanced risk analysis with AI insights
                risk_analysis = self._enhance_risk_analysis_with_ai(risk_analysis, ai_response)
        
        # CRITICAL: Make risk analysis serializable
        state["risk_assessment"] = make_serializable(risk_analysis)
        
        # Determine if risks are acceptable
        high_risk_items = risk_analysis.get("high_risk_items", [])
        overall_risk = risk_analysis.get("overall_risk_score", 0.3)
        
        print(f"{self.agent_name}: Risk score: {overall_risk}, High risk items: {len(high_risk_items)}")
        
        if overall_risk > 0.7 or len(high_risk_items) > 2:
            print(f"{self.agent_name}: High risk detected - requiring human review")
            state["requires_human_review"] = True
            state["next_action"] = "human_review"
        else:
            state["next_action"] = "validation"
            
        state["current_step"] = "risk_assessment"
        
        return state
    
    def _perform_quantitative_risk_analysis(self, state: SupplyChainState) -> Dict[str, Any]:
        """Perform quantitative risk analysis on results"""
        try:
            risk_metrics = {
                "overall_risk_score": 0.3,  # Default low risk
                "high_risk_items": [],
                "stockout_probability": {},
                "overstock_risk": {},
                "financial_impact": {
                    "potential_savings": 50000.0,
                    "risk_exposure": 15000.0
                },
                "risk_level": "LOW"
            }
            
            # Load and analyze results if available
            if self.config.INVENTORY_OPTIMIZATION_RESULTS.exists():
                try:
                    df = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_RESULTS)
                    
                    # Calculate risk metrics from actual data
                    high_risk_items = []
                    stockout_prob = {}
                    overstock_risk = {}
                    
                    for _, row in df.iterrows():
                        sku = row.get('SKU', str(row.get('item_id', 'unknown')))
                        forecasted_demand = float(row.get('forecasted_demand', 0))
                        safety_stock = float(row.get('safety_stock', 0))
                        reorder_point = float(row.get('reorder_point', 0))
                        
                        # Calculate stockout probability (simplified)
                        if safety_stock < forecasted_demand * 0.1:  # Less than 10% buffer
                            stockout_prob[sku] = 0.15  # 15% risk
                            if forecasted_demand > 1000:  # High volume item
                                high_risk_items.append(sku)
                        else:
                            stockout_prob[sku] = 0.05  # 5% risk
                        
                        # Calculate overstock risk
                        if safety_stock > forecasted_demand * 0.5:  # More than 50% buffer
                            overstock_risk[sku] = 0.20  # 20% risk
                        else:
                            overstock_risk[sku] = 0.05  # 5% risk
                    
                    # Update risk metrics
                    risk_metrics.update({
                        "stockout_probability": stockout_prob,
                        "overstock_risk": overstock_risk,
                        "high_risk_items": high_risk_items
                    })
                    
                    # Calculate overall risk score
                    avg_stockout_risk = sum(stockout_prob.values()) / len(stockout_prob) if stockout_prob else 0.05
                    avg_overstock_risk = sum(overstock_risk.values()) / len(overstock_risk) if overstock_risk else 0.05
                    overall_risk = (avg_stockout_risk + avg_overstock_risk) / 2
                    
                    risk_metrics["overall_risk_score"] = float(overall_risk)
                    
                    # Classify risk level
                    if overall_risk < 0.3:
                        risk_metrics["risk_level"] = "LOW"
                    elif overall_risk < 0.7:
                        risk_metrics["risk_level"] = "MEDIUM"
                    else:
                        risk_metrics["risk_level"] = "HIGH"
                    
                except Exception as e:
                    print(f"{self.agent_name}: Error analyzing optimization results: {e}")
                    risk_metrics["analysis_error"] = str(e)
                    
            return make_serializable(risk_metrics)
            
        except Exception as e:
            print(f"{self.agent_name}: Error in quantitative analysis: {e}")
            return make_serializable({
                "overall_risk_score": 0.5,  # Medium risk due to error
                "error": str(e),
                "high_risk_items": ["Analysis failed"],
                "risk_level": "MEDIUM"
            })
    
    def _enhance_risk_analysis_with_ai(self, base_analysis: Dict[str, Any], ai_response: str) -> Dict[str, Any]:
        """Enhance risk analysis with AI insights"""
        try:
            # Extract risk classification from AI response
            if "HIGH" in ai_response.upper():
                base_analysis["ai_risk_classification"] = "HIGH"
                base_analysis["overall_risk_score"] = max(base_analysis.get("overall_risk_score", 0.3), 0.7)
            elif "MEDIUM" in ai_response.upper():
                base_analysis["ai_risk_classification"] = "MEDIUM"
                base_analysis["overall_risk_score"] = max(base_analysis.get("overall_risk_score", 0.3), 0.5)
            else:
                base_analysis["ai_risk_classification"] = "LOW"
            
            # Add AI insights
            base_analysis["ai_insights"] = ai_response
            
            # Extract mitigation strategies if present
            if "mitigation" in ai_response.lower():
                base_analysis["ai_mitigation_strategies"] = "See AI insights for detailed mitigation strategies"
            
            return make_serializable(base_analysis)
            
        except Exception as e:
            print(f"{self.agent_name}: Error enhancing analysis with AI: {e}")
            return base_analysis