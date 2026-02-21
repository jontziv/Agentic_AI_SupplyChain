from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
from config import SupplyChainConfig
from state import SupplyChainState, make_serializable

from datetime import datetime
from state import make_serializable
from agents.tooling import tools_system_message, log_agent_action


class ValidatorAgent:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        self.agent_name = "Validator"
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
    
    def validate_recommendations(self, state: SupplyChainState) -> SupplyChainState:
        """Final validation of all recommendations before human review"""
        
        print(f"{self.agent_name}: Starting validation of recommendations...")
        
        # Perform validation analysis first
        validation_results = self._perform_validation_checks(state)
        log_agent_action(state, self.agent_name, "compute_stats", {"source": "validation_checks"}, result_preview=validation_results)
        
        # Try AI validation if LLM is available
        ai_validation = "REVIEW"
        if self.llm:
            print(f"{self.agent_name}: Running AI-powered validation...")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a senior supply chain validation specialist.
                Perform final validation of all recommendations:
                1. Cross-check forecast vs optimization consistency
                2. Validate against business rules and constraints
                3. Verify mathematical accuracy of calculations
                4. Check for logical inconsistencies
                5. Assess implementation feasibility
                6. Generate executive summary with key metrics
                
                Provide a PASS/FAIL/REVIEW validation with detailed reasoning.
                Focus on practical implementability and business impact."""),
                SystemMessage(content=tools_system_message(self.config)),
                HumanMessage(content=f"""
                Complete Analysis Results:
                - Demand Forecast: {state.get('demand_forecast', {})}
                - Inventory Optimization: {state.get('inventory_optimization', {})}
                - Risk Assessment: {state.get('risk_assessment', {})}
                - Business Rules: Service Level {self.config.SERVICE_LEVEL_TARGET}
                - Validation Checks: {validation_results}
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
                
                # Extract validation decision
                if "PASS" in ai_response.upper():
                    ai_validation = "PASS"
                elif "FAIL" in ai_response.upper():
                    ai_validation = "FAIL"
                else:
                    ai_validation = "REVIEW"
                
                validation_results["ai_validation"] = ai_response
        
        validation_results["overall_validation"] = ai_validation
        
        # Create final recommendations package
        final_recommendations = self._create_final_recommendations(state, validation_results)
        state["recommendations"] = make_serializable(final_recommendations)
        
        # Create comprehensive summary
        summary_data = self._create_comprehensive_summary(state)
        
        # Store final recommendations with summary
        state["final_recommendations"] = make_serializable({
            "recommendations": final_recommendations,
            "summary": summary_data,
            "validation": validation_results,
            "ai_analysis": validation_results.get("ai_validation", "Analysis completed"),
            "risk_assessment": state.get("risk_assessment", {})
        })
        
        print(f"{self.agent_name}: Validation complete - {ai_validation}")
        
        # Always require human review for final approval
        state["requires_human_review"] = True
        state["next_action"] = "human_review"
        state["current_step"] = "validation"
        
        return state
    
    def _perform_validation_checks(self, state: SupplyChainState) -> Dict[str, Any]:
        """Perform systematic validation checks"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
            "issues_found": [],
            "validation_score": 0.0
        }
        
        try:
            score = 1.0
            
            # Check 1: Demand forecast validation
            validation["checks_performed"].append("demand_forecast_validation")
            demand_forecast = state.get("demand_forecast", {})
            if demand_forecast.get("status") == "success":
                confidence = float(demand_forecast.get("confidence_score", 0))
                if confidence < 0.6:
                    validation["issues_found"].append("Low forecast confidence")
                    score -= 0.2
            else:
                validation["issues_found"].append("Demand forecast failed")
                score -= 0.4
            
            # Check 2: Inventory optimization validation
            validation["checks_performed"].append("inventory_optimization_validation")
            inventory_opt = state.get("inventory_optimization", {})
            if inventory_opt.get("status") == "success":
                opt_score = float(inventory_opt.get("optimization_score", 0))
                if opt_score < 0.7:
                    validation["issues_found"].append("Low optimization quality")
                    score -= 0.2
            else:
                validation["issues_found"].append("Inventory optimization failed")
                score -= 0.4
            
            # Check 3: Risk assessment validation
            validation["checks_performed"].append("risk_assessment_validation")
            risk_data = state.get("risk_assessment", {})
            overall_risk = float(risk_data.get("overall_risk_score", 0.3))
            if overall_risk > 0.8:
                validation["issues_found"].append("Very high risk score")
                score -= 0.3
            elif overall_risk > 0.6:
                validation["issues_found"].append("Elevated risk score")
                score -= 0.1
            
            # Check 4: Data consistency
            validation["checks_performed"].append("data_consistency_check")
            if self._check_data_consistency():
                validation["issues_found"].append("Data consistency issues detected")
                score -= 0.1
            
            # Check 5: Business rules compliance
            validation["checks_performed"].append("business_rules_check")
            if not self._check_business_rules_compliance(state):
                validation["issues_found"].append("Business rules compliance issues")
                score -= 0.2
            
            validation["validation_score"] = max(0.0, score)
            
            return make_serializable(validation)
            
        except Exception as e:
            print(f"{self.agent_name}: Error in validation checks: {e}")
            validation["issues_found"].append(f"Validation error: {str(e)}")
            validation["validation_score"] = 0.5
            return make_serializable(validation)
    
    def _check_data_consistency(self) -> bool:
        """Check consistency between forecast and optimization data"""
        try:
            # Check if both result files exist and have consistent SKUs
            if (self.config.DEMAND_FORECAST_RESULTS.exists() and 
                self.config.INVENTORY_OPTIMIZATION_RESULTS.exists()):
                
                df_forecast = pd.read_csv(self.config.DEMAND_FORECAST_RESULTS)
                df_inventory = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_RESULTS)
                
                forecast_skus = set(df_forecast['SKU'].unique())
                inventory_skus = set(df_inventory['SKU'].unique())
                
                # Check if SKUs match
                if forecast_skus != inventory_skus:
                    return True  # Inconsistency found
            
            return False  # No inconsistency
            
        except Exception as e:
            print(f"{self.agent_name}: Error checking data consistency: {e}")
            return True  # Assume inconsistency on error
    
    def _check_business_rules_compliance(self, state: SupplyChainState) -> bool:
        """Check compliance with business rules"""
        try:
            # Check service level compliance
            target_service_level = self.config.SERVICE_LEVEL_TARGET
            
            # Check if optimization results meet service level requirements
            if self.config.INVENTORY_OPTIMIZATION_RESULTS.exists():
                df = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_RESULTS)
                
                # Basic validation - check if safety stock is reasonable
                if 'safety_stock' in df.columns and 'forecasted_demand' in df.columns:
                    for _, row in df.iterrows():
                        safety_stock = float(row.get('safety_stock', 0))
                        forecasted_demand = float(row.get('forecasted_demand', 1))
                        
                        # Safety stock should be reasonable (5-50% of monthly demand)
                        safety_ratio = safety_stock / forecasted_demand if forecasted_demand > 0 else 0
                        if safety_ratio > 0.5 or safety_ratio < 0.05:
                            return False  # Business rule violation
            
            return True  # Compliant
            
        except Exception as e:
            print(f"{self.agent_name}: Error checking business rules: {e}")
            return False  # Assume non-compliance on error
    
    def _create_final_recommendations(self, state: SupplyChainState, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create structured final recommendations"""
        
        recommendations = []
        
        # Demand forecast recommendation
        demand_forecast = state.get("demand_forecast", {})
        recommendations.append(make_serializable({
            "type": "demand_forecast",
            "summary": "AI-optimized demand forecasting completed",
            "confidence": float(demand_forecast.get("confidence_score", 0.8)),
            "validation": validation_results.get("overall_validation", "REVIEW"),
            "status": demand_forecast.get("status", "unknown"),
            "details": demand_forecast.get("output", "No details available")
        }))
        
        # Inventory optimization recommendation
        inventory_opt = state.get("inventory_optimization", {})
        recommendations.append(make_serializable({
            "type": "inventory_optimization",
            "summary": "Inventory parameters optimized for target service level",
            "confidence": float(inventory_opt.get("optimization_score", 0.8)),
            "validation": validation_results.get("overall_validation", "REVIEW"),
            "status": inventory_opt.get("status", "unknown"),
            "details": inventory_opt.get("output", "No details available")
        }))
        
        # Risk assessment recommendation
        risk_data = state.get("risk_assessment", {})
        risk_score = float(risk_data.get("overall_risk_score", 0.3))
        recommendations.append(make_serializable({
            "type": "risk_assessment",
            "summary": f"Overall risk level: {risk_data.get('risk_level', 'UNKNOWN')} (score: {risk_score:.2f})",
            "confidence": 0.9,
            "validation": validation_results.get("overall_validation", "REVIEW"),
            "risk_score": risk_score,
            "high_risk_items": risk_data.get("high_risk_items", [])
        }))
        
        return recommendations
    
    def _create_comprehensive_summary(self, state: SupplyChainState) -> Dict[str, Any]:
        """Create comprehensive summary for final recommendations"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_items": 0,
            "avg_reorder_point": None,
            "avg_safety_stock": None,
            "avg_eoq": None
        }
        
        try:
            if self.config.INVENTORY_OPTIMIZATION_RESULTS.exists():
                df = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_RESULTS)
                
                summary["total_items"] = len(df)
                
                if 'reorder_point' in df.columns:
                    summary["avg_reorder_point"] = float(df['reorder_point'].mean())
                
                if 'safety_stock' in df.columns:
                    summary["avg_safety_stock"] = float(df['safety_stock'].mean())
                
                if 'eoq' in df.columns:
                    summary["avg_eoq"] = float(df['eoq'].mean())
            
            return make_serializable(summary)
            
        except Exception as e:
            print(f"{self.agent_name}: Error creating summary: {e}")
            return make_serializable(summary)