import app as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from config import SupplyChainConfig
from state import SupplyChainState

class HumanInTheLoopInterface:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
    
    def render_human_review_interface(self, state: SupplyChainState) -> Dict[str, Any]:
        """Render comprehensive human review interface"""
        
        st.header("🧑‍💼 Human Review Required")
        
        # Executive Summary
        self._render_executive_summary(state)
        
        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Analysis", 
            "📈 Demand Forecast", 
            "📦 Inventory Optimization", 
            "⚠️ Risk Assessment"
        ])
        
        with tab1:
            self._render_data_analysis(state)
        
        with tab2:
            self._render_demand_forecast(state)
        
        with tab3:
            self._render_inventory_optimization(state)
        
        with tab4:
            self._render_risk_assessment(state)
        
        with st.expander("🛠 Agent Actions (ReAct log)"):
            actions = state.get("agent_actions", [])
            if actions:
                for a in actions[-50:]:
                    st.write(f"**{a.get('ts')} — {a.get('agent')}** → `{a.get('action')}` {a.get('args')}")
                    if a.get("result_preview"):
                        st.code(str(a.get('result_preview'))[:2000])
            else:
                st.caption("No actions recorded.")

        # Decision Interface
        return self._render_decision_interface(state)
    
    def _render_executive_summary(self, state: SupplyChainState):
        """Render executive summary dashboard"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            forecast_confidence = state.get("demand_forecast", {}).get("confidence_score", 0)
            st.metric("Forecast Confidence", f"{forecast_confidence*100:.1f}%")
        
        with col2:
            risk_score = state.get("risk_assessment", {}).get("overall_risk_score", 0)
            st.metric("Risk Score", f"{risk_score:.2f}", delta="Low" if risk_score < 0.5 else "High")
        
        with col3:
            optimization_score = state.get("inventory_optimization", {}).get("optimization_score", 0)
            st.metric("Optimization Score", f"{optimization_score*100:.1f}%")
        
        with col4:
            service_level = self.config.SERVICE_LEVEL_TARGET
            st.metric("Target Service Level", f"{service_level*100:.1f}%")
        
        # Key Insights
        st.subheader("🎯 Key Insights")
        recommendations = state.get("recommendations", [])
        for rec in recommendations[:3]:  # Show top 3
            confidence = rec.get("confidence", 0)
            color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            st.write(f"{color} **{rec.get('type', 'Unknown')}**: {rec.get('summary', '')}")
    
    def _render_data_analysis(self, state: SupplyChainState):
        """Render data analysis results"""
        
        st.subheader("Data Quality Assessment")
        
        issues = state.get("data_quality_issues", [])
        if issues:
            st.warning("Data Quality Issues Found:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("No significant data quality issues detected")
        
        # Seasonal factors
        seasonal_factors = state.get("seasonal_factors", {})
        if seasonal_factors:
            st.subheader("Seasonal Adjustment Factors")
            st.json(seasonal_factors)
    
    def _render_demand_forecast(self, state: SupplyChainState):
        """Render demand forecast results"""
        
        forecast = state.get("demand_forecast", {})
        
        if forecast.get("status") == "success":
            st.success("Demand forecasting completed successfully")
            
            confidence = forecast.get("confidence_score", 0)
            st.metric("Forecast Confidence", f"{confidence*100:.1f}%")
            
            # Show AI strategy
            if "ai_strategy" in forecast:
                st.subheader("AI Forecasting Strategy")
                st.text_area("Strategy Details", forecast["ai_strategy"], height=150)
        else:
            st.error(f"Forecast failed: {forecast.get('error', 'Unknown error')}")
    
    def _render_inventory_optimization(self, state: SupplyChainState):
        """Render inventory optimization results"""
        
        optimization = state.get("inventory_optimization", {})
        
        if optimization.get("status") == "success":
            st.success("Inventory optimization completed successfully")
            
            score = optimization.get("optimization_score", 0)
            st.metric("Optimization Score", f"{score*100:.1f}%")
            
            # Show results data if available
            if self.config.INVENTORY_OPTIMIZATION_RESULTS.exists():
                try:
                    df = pd.read_csv(self.config.INVENTORY_OPTIMIZATION_RESULTS)
                    st.subheader("Optimization Results")
                    st.dataframe(df.head(10))
                except:
                    st.warning("Could not load results data")
        else:
            st.error(f"Optimization failed: {optimization.get('error', 'Unknown error')}")
    
    def _render_risk_assessment(self, state: SupplyChainState):
        """Render risk assessment results"""
        
        risk_assessment = state.get("risk_assessment", {})
        
        overall_risk = risk_assessment.get("overall_risk_score", 0)
        risk_level = "LOW" if overall_risk < 0.3 else "MEDIUM" if overall_risk < 0.7 else "HIGH"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Risk Level", risk_level)
        with col2:
            st.metric("Risk Score", f"{overall_risk:.2f}")
        
        # High risk items
        high_risk_items = risk_assessment.get("high_risk_items", [])
        if high_risk_items:
            st.warning("High Risk Items:")
            for item in high_risk_items:
                st.write(f"- {item}")
        
        # Financial impact
        financial_impact = risk_assessment.get("financial_impact", {})
        if financial_impact:
            st.subheader("Financial Impact Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Potential Savings", f"${financial_impact.get('potential_savings', 0):,.0f}")
            with col2:
                st.metric("Risk Exposure", f"${financial_impact.get('risk_exposure', 0):,.0f}")
    
    def _render_decision_interface(self, state: SupplyChainState) -> Dict[str, Any]:
        """Render human decision interface"""
        
        st.header("📋 Decision Required")
        
        # Decision options
        decision = st.radio(
            "What would you like to do?",
            [
                "✅ Approve all recommendations",
                "✏️ Approve with modifications", 
                "🔄 Request re-analysis",
                "❌ Reject recommendations"
            ]
        )
        
        feedback = {}
        
        if "modifications" in decision:
            st.subheader("Specify Modifications")
            
            col1, col2 = st.columns(2)
            with col1:
                safety_stock_adjustment = st.slider(
                    "Safety Stock Adjustment (%)", -50, 100, 0
                )
                service_level_override = st.slider(
                    "Service Level Override", 0.80, 0.99, self.config.SERVICE_LEVEL_TARGET
                )
            
            with col2:
                reorder_point_adjustment = st.slider(
                    "Reorder Point Adjustment (%)", -30, 50, 0
                )
                
            feedback["modifications"] = {
                "safety_stock_adjustment": safety_stock_adjustment,
                "service_level_override": service_level_override,
                "reorder_point_adjustment": reorder_point_adjustment
            }
        
        elif "re-analysis" in decision:
            st.subheader("Re-analysis Requirements")
            reanalysis_reason = st.text_area(
                "Why is re-analysis needed?",
                placeholder="Please specify what should be changed in the analysis..."
            )
            feedback["reanalysis_reason"] = reanalysis_reason
        
        elif "Reject" in decision:
            st.subheader("Rejection Reasons")
            rejection_reasons = st.multiselect(
                "Select rejection reasons:",
                [
                    "Forecast accuracy too low",
                    "Risk level too high", 
                    "Business constraints not met",
                    "Implementation not feasible",
                    "Data quality issues",
                    "Other"
                ]
            )
            
            other_reason = ""
            if "Other" in rejection_reasons:
                other_reason = st.text_area("Specify other reason:")
            
            feedback["rejection_reasons"] = rejection_reasons
            feedback["other_reason"] = other_reason
        
        # Comments
        comments = st.text_area(
            "Additional Comments",
            placeholder="Any additional feedback or instructions..."
        )
        feedback["comments"] = comments
        
        # Submit decision
        if st.button("Submit Decision", type="primary"):
            feedback.update({
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
                "user": "supply_planner"  # Could be actual user ID
            })
            
            return feedback
        
        return {}