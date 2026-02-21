import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List

from config import SupplyChainConfig, WorkflowStatus
from workflow import SupplyChainWorkflowEngine, WorkflowState


# =========================
# Page configuration & THEME
# =========================
st.set_page_config(
    page_title="AI Supply Chain Optimization",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -- Futuristic Agentic CSS with improved agent log styling
st.markdown("""
<style>
:root{
  --accent: #8b5cf6;
  --accent-2: #06b6d4;
  --accent-3: #10b981;
  --good: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
  --muted: #94a3b8;
  --bg: rgba(15,23,42,0.8);
  --glass: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.12);
  --text: #e2e8f0;
  --text-muted: #94a3b8;
}

/* Dark theme background */
.stApp {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  color: var(--text);
}

/* Hide default header */
header[data-testid="stHeader"] { 
  background: transparent !important;
  border: none !important;
}

/* Custom containers */
.block-container { 
  padding-top: 0.5rem; 
  max-width: 95%;
}

/* Hero section */
.hero-container {
  background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(6,182,212,0.15));
  border: 1px solid var(--border); 
  border-radius: 20px; 
  padding: 24px 28px; 
  margin-bottom: 20px;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
}

.hero-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
}

/* Status indicators */
.status-badge {
  display: inline-flex; 
  align-items: center; 
  gap: 8px; 
  font-size: 0.9rem; 
  padding: 8px 16px; 
  border-radius: 20px; 
  border: 1px solid var(--border); 
  background: var(--bg);
  backdrop-filter: blur(8px);
}

.status-dot { 
  width: 10px; 
  height: 10px; 
  border-radius: 50%; 
  display: inline-block; 
}
.dot.ok { background: var(--good); box-shadow: 0 0 8px rgba(34,197,94,0.4); }
.dot.warn { background: var(--warn); box-shadow: 0 0 8px rgba(245,158,11,0.4); }
.dot.bad { background: var(--bad); box-shadow: 0 0 8px rgba(239,68,68,0.4); }
.dot.processing { 
  background: var(--accent); 
  box-shadow: 0 0 8px rgba(139,92,246,0.4);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Process steps */
.process-steps {
  display: flex !important;
  gap: 12px !important;
  flex-wrap: wrap !important;
  margin: 16px 0 !important;
  align-items: center !important;
}

.step-chip {
  display: inline-flex !important; 
  align-items: center !important; 
  gap: 8px !important; 
  padding: 10px 16px !important; 
  border-radius: 20px !important;
  border: 1px solid rgba(255,255,255,0.12) !important; 
  background: rgba(15,23,42,0.8) !important; 
  color: #94a3b8 !important;
  font-size: 0.85rem !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(8px) !important;
  white-space: nowrap !important;
}

.step-chip .step-dot { 
  width: 8px !important; 
  height: 8px !important; 
  border-radius: 50% !important; 
  background: #475569 !important;
  display: inline-block !important;
}

.step-chip.active { 
  border-color: rgba(139,92,246,0.8) !important; 
  background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(6,182,212,0.15)) !important; 
  color: #e2e8f0 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(139,92,246,0.2) !important;
}

.step-chip.active .step-dot { 
  background: #8b5cf6 !important;
  box-shadow: 0 0 6px rgba(139,92,246,0.6) !important;
}

/* Metric cards */
.metric-card {
  background: var(--glass); 
  border: 1px solid var(--border); 
  border-radius: 16px; 
  padding: 20px;
  backdrop-filter: blur(12px);
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: rgba(139,92,246,0.4);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

.metric-card h4 { 
  margin: 0 0 12px; 
  font-weight: 600; 
  color: var(--text-muted);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-card .value { 
  font-size: 2rem; 
  font-weight: 800; 
  color: var(--text);
  line-height: 1;
}

/* Enhanced agent log styling */
.agent-log-container {
  background: #0a0f1c; 
  border-radius: 16px; 
  border: 1px solid #1e293b; 
  overflow: hidden;
  max-height: 600px;
  display: flex;
  flex-direction: column;
}

.agent-log-header {
  background: linear-gradient(135deg, #1e293b, #334155);
  padding: 16px 20px;
  border-bottom: 1px solid #334155;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.agent-log-header h3 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #e2e8f0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.log-terminal {
  font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #e2e8f0; 
  overflow-y: auto;
  flex: 1;
  padding: 0;
}

.log-entry {
  padding: 12px 20px;
  border-bottom: 1px solid rgba(51,65,85,0.3);
  display: grid;
  grid-template-columns: 80px 100px 1fr;
  gap: 16px;
  align-items: flex-start;
  transition: background-color 0.2s ease;
}

.log-entry:hover {
  background-color: rgba(51,65,85,0.2);
}

.log-timestamp {
  color: #64748b;
  font-size: 0.8rem;
  white-space: nowrap;
}

.log-agent {
  color: #a5b4fc;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
}

.log-content {
  min-width: 0;
}

.log-action {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.action-type {
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 6px;
  border: 1px solid #334155;
  background: #1e293b;
  color: #94a3b8;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.action-type.thought { 
  color: #c4b5fd; 
  border-color: #7c3aed; 
  background: rgba(124,58,237,0.1); 
}

.action-type.action { 
  color: #86efac; 
  border-color: #059669; 
  background: rgba(5,150,105,0.1); 
}

.action-type.observation { 
  color: #fca5a5; 
  border-color: #dc2626; 
  background: rgba(220,38,38,0.1); 
}

.action-name {
  color: #e2e8f0;
  font-weight: 500;
}

.log-details {
  font-size: 0.85rem;
  color: #cbd5e1;
  margin-top: 4px;
  word-break: break-word;
}

.log-result {
  background: rgba(15,23,42,0.6);
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 8px 12px;
  margin-top: 8px;
  font-size: 0.8rem;
  color: #94a3b8;
  max-height: 120px;
  overflow-y: auto;
}

/* Decision panel */
.decision-panel {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  backdrop-filter: blur(12px);
}

.decision-panel h3 {
  margin: 0 0 16px;
  color: var(--text);
  font-size: 1.2rem;
  font-weight: 600;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 600;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(139,92,246,0.3);
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(139,92,246,0.4);
}

/* Tabs */
.stTabs > div > div > div > div {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  backdrop-filter: blur(12px);
}

/* Dividers */
.divider { 
  border: 0; 
  border-top: 1px solid var(--border); 
  margin: 20px 0; 
  opacity: 0.6;
}

/* Success/Warning/Error messages */
.stAlert {
  border-radius: 12px;
  backdrop-filter: blur(12px);
}

/* DataFrames */
.stDataFrame {
  border-radius: 12px;
  overflow: hidden;
}

/* Progress indicators */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Cached config + engine
# =========================
@st.cache_resource
def get_config():
    return SupplyChainConfig()

@st.cache_resource
def get_workflow_engine():
    cfg = get_config()
    return SupplyChainWorkflowEngine(cfg)

config = get_config()
workflow_engine = get_workflow_engine()


# =========================
# Session State
# =========================
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = WorkflowState()
if 'workflow_started' not in st.session_state:
    st.session_state.workflow_started = False


# =========================
# Helpers
# =========================
def reset_workflow():
    st.session_state.workflow_state = WorkflowState()
    st.session_state.workflow_started = False
    workflow_engine.state = WorkflowState()

def start_workflow():
    with st.spinner("🚀 Initializing AI agents..."):
        state = workflow_engine.start_workflow()
        st.session_state.workflow_state = state
        st.session_state.workflow_started = True
    return state

def process_human_decision(feedback: Dict[str, Any]):
    with st.spinner("Processing your decision..."):
        state = workflow_engine.process_human_feedback(feedback)
        st.session_state.workflow_state = state
    return state

def get_status_info(status: WorkflowStatus):
    """Get status display information"""
    status_map = {
        WorkflowStatus.IDLE: ("READY", "ok", "System ready to begin optimization"),
        WorkflowStatus.DATA_ANALYSIS: ("ANALYZING", "processing", "AI agents analyzing data quality"),
        WorkflowStatus.DEMAND_FORECASTING: ("FORECASTING", "processing", "Running demand prediction models"),
        WorkflowStatus.INVENTORY_OPTIMIZATION: ("OPTIMIZING", "processing", "Calculating optimal parameters"),
        WorkflowStatus.HUMAN_REVIEW: ("REVIEW REQUIRED", "warn", "Human review needed"),
        WorkflowStatus.APPROVED: ("APPROVED", "ok", "Recommendations approved"),
        WorkflowStatus.REJECTED: ("REJECTED", "bad", "Recommendations rejected"),
        WorkflowStatus.ERROR: ("ERROR", "bad", "System error occurred")
    }
    return status_map.get(status, ("UNKNOWN", "warn", "Unknown status"))

def render_process_steps():
    """Render modern process step indicators"""
    state = st.session_state.workflow_state
    
    steps = [
        ("Data Analysis", "data_analysis"),
        ("AI Forecasting", "forecasting"),
        ("Optimization", "optimizing"),
        ("Risk Assessment", "risk_assessment"),
        ("Validation", "validation"),
        ("Human Review", "human_review")
    ]
    
    current_step = getattr(state, "current_step", "idle")
    
    # Create step chips using columns for better layout
    cols = st.columns(len(steps))
    
    for i, (label, key) in enumerate(steps):
        is_active = current_step == key or (current_step == "human_review" and key == "validation")
        
        with cols[i]:
            if is_active:
                st.markdown(f'''
                <div class="step-chip active">
                    <span class="step-dot"></span>
                    {label}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="step-chip">
                    <span class="step-dot"></span>
                    {label}
                </div>
                ''', unsafe_allow_html=True)


# =========================
# Enhanced Agent Log Component
# =========================
def render_enhanced_agent_log(state: WorkflowState):
    """Render enhanced agent log with improved ReAct-style display"""
    actions = getattr(state, "agent_actions", [])
    
    if not actions:
        # Use Streamlit components for empty state
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #64748b; background: #0a0f1c; border-radius: 12px; border: 1px solid #1e293b;">
            <div style="font-size: 2rem; margin-bottom: 12px;">🔍</div>
            <div style="font-size: 1.1rem;">No agent activity yet</div>
            <div style="font-size: 0.9rem; margin-top: 8px; opacity: 0.8;">Start the workflow to see agents in action</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Process and deduplicate actions
    processed_actions = []
    seen_actions = set()
    
    for action in actions:
        action_name = action.get('action', '')
        timestamp = action.get('ts', '')
        agent = action.get('agent', '')
        
        # Create a unique key for deduplication
        unique_key = f"{agent}_{action_name}_{timestamp}"
        
        # Skip if we've already seen this exact action
        if unique_key in seen_actions:
            continue
        seen_actions.add(unique_key)
        
        # Skip redundant INFO entries that just echo actions
        if action_name.lower().endswith('info') and len(action_name) > 4:
            base_action = action_name[:-4].lower()  # Remove 'info' suffix
            if any(a.get('action', '').lower() == base_action for a in actions):
                continue
        
        processed_actions.append(action)
    
    # Build the log HTML using components.html for better rendering
    log_html = """
    <div style="background: #0a0f1c; border-radius: 16px; border: 1px solid #1e293b; overflow: hidden; max-height: 600px; font-family: 'JetBrains Mono', monospace;">
        <div style="background: linear-gradient(135deg, #1e293b, #334155); padding: 16px 20px; border-bottom: 1px solid #334155;">
            <div style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem;">🤖 Agent Activity Monitor</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Real-time agent actions</div>
        </div>
        <div style="overflow-y: auto; max-height: 500px;">
    """
    
    # Display processed actions
    for action in processed_actions[-20:]:  # Show last 20 actions
        timestamp = action.get('ts', '')
        agent = action.get('agent', 'System')
        action_name = action.get('action', '')
        args = action.get('args', {})
        result_preview = action.get('result_preview', '')
        status = action.get('status', 'completed')
        
        # Format timestamp
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M:%S")
            else:
                time_str = "--:--:--"
        except:
            time_str = str(timestamp)[:8] if timestamp else "--:--:--"
        
        # Determine action type and styling
        action_type = "action"
        type_color = "#86efac"
        type_bg = "rgba(5,150,105,0.1)"
        type_border = "#059669"
        
        if "think" in action_name.lower() or "reason" in action_name.lower():
            action_type = "thought"
            type_color = "#c4b5fd"
            type_bg = "rgba(124,58,237,0.1)"
            type_border = "#7c3aed"
        elif "result" in action_name.lower() or "output" in action_name.lower():
            action_type = "observation"
            type_color = "#fca5a5"
            type_bg = "rgba(220,38,38,0.1)"
            type_border = "#dc2626"
        
        # Format action arguments
        args_str = ""
        if args and isinstance(args, dict):
            key_args = []
            for k, v in list(args.items())[:2]:  # Show first 2 args
                if isinstance(v, str) and len(v) > 30:
                    v = v[:30] + "..."
                key_args.append(f"{k}={v}")
            if key_args:
                args_str = f"({', '.join(key_args)})"
        
        # Format result preview
        result_str = ""
        if result_preview and str(result_preview).strip():
            result_text = str(result_preview)
            if len(result_text) > 200:
                result_text = result_text[:200] + "..."
            result_str = f'''
            <div style="background: rgba(15,23,42,0.6); border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-top: 8px; font-size: 0.8rem; color: #94a3b8; max-height: 120px; overflow-y: auto;">
                {result_text}
            </div>
            '''
        
        log_html += f'''
        <div style="padding: 12px 20px; border-bottom: 1px solid rgba(51,65,85,0.3); display: grid; grid-template-columns: 80px 100px 1fr; gap: 16px; align-items: flex-start;">
            <div style="color: #64748b; font-size: 0.8rem; white-space: nowrap;">{time_str}</div>
            <div style="color: #a5b4fc; font-size: 0.8rem; font-weight: 600; white-space: nowrap;">{agent}</div>
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 0.7rem; padding: 2px 8px; border-radius: 6px; border: 1px solid {type_border}; background: {type_bg}; color: {type_color}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                        {action_type.upper()}
                    </span>
                    <span style="color: #e2e8f0; font-weight: 500;">{action_name}</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px; word-break: break-word;">{args_str}</div>
                {result_str}
            </div>
        </div>
        '''
    
    log_html += """
        </div>
    </div>
    """
    
    # Use components.html for proper rendering
    components.html(log_html, height=620, scrolling=True)


# =========================
# Data Display Components
# =========================
def render_data_preview():
    """Render data preview section"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Demand Forecast Data")
        try:
            if config.DEMAND_FORECAST_FILE.exists():
                df = pd.read_csv(config.DEMAND_FORECAST_FILE)
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Total records: {len(df):,}")
            else:
                st.error("Demand forecast file not found")
        except Exception as e:
            st.error(f"Error loading demand data: {e}")
    
    with col2:
        st.subheader("📦 Inventory Data")
        try:
            if config.INVENTORY_OPTIMIZATION_FILE.exists():
                df = pd.read_csv(config.INVENTORY_OPTIMIZATION_FILE)
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Total records: {len(df):,}")
            else:
                st.error("Inventory optimization file not found")
        except Exception as e:
            st.error(f"Error loading inventory data: {e}")

def render_results_visualization():
    """Render results with enhanced visualizations"""
    results_data = workflow_engine.get_results_data()
    
    if 'demand_forecast' in results_data:
        st.subheader("📊 Demand Forecast Results")
        df = results_data['demand_forecast']
        st.dataframe(df.head(10), use_container_width=True)
        
        # Enhanced visualization
        if len(df) > 0 and 'forecasted_demand' in df.columns:
            fig = px.bar(df.head(10), x='SKU', y='forecasted_demand', 
                        title="Forecasted Demand by SKU",
                        color='forecasted_demand',
                        color_continuous_scale='viridis')
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_ai_analysis_with_decision():
    """Render AI analysis with decision panel and agent log"""
    state = st.session_state.workflow_state
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧠 AI Analysis & Insights")
        
        # AI Analysis Content
        if state.final_recommendations and isinstance(state.final_recommendations, dict):
            ai_analysis = state.final_recommendations.get('ai_analysis')
            if isinstance(ai_analysis, dict) and ai_analysis.get('status') == 'success':
                st.markdown(ai_analysis.get('analysis', ''))
            elif isinstance(ai_analysis, str):
                st.markdown(ai_analysis)
            else:
                st.info("🔄 AI analysis will be available after workflow completion.")
        else:
            st.info("🔄 AI analysis will be available after workflow completion.")
        
        # Risk Assessment
        if state.final_recommendations and isinstance(state.final_recommendations, dict):
            risk = state.final_recommendations.get('risk_assessment')
            if risk:
                st.subheader("⚠️ Risk Assessment")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    risk_level = risk.get('risk_level', '-')
                    st.metric("Risk Level", risk_level)
                with col_r2:
                    risk_score = risk.get('overall_score', 0.0)
                    st.metric("Risk Score", f"{risk_score:.2f}")
                
                high_risk_items = risk.get('high_risk_items', [])
                if high_risk_items:
                    st.warning("**High Risk Items:**")
                    for item in high_risk_items[:5]:
                        st.write(f"• {item}")
    
    with col2:
        # Decision Panel (only show when review is required)
        if getattr(state, "requires_human_review", False):
            st.markdown('<div class="decision-panel">', unsafe_allow_html=True)
            st.markdown("### 🎯 Decision Required")
            
            decision = st.radio(
                "Choose your action:",
                ["✅ Approve all recommendations", "❌ Reject recommendations"],
                index=0
            )
            
            comments = st.text_area(
                "Comments (optional)", 
                placeholder="Any feedback or notes...",
                height=100
            )
            
            if st.button("Submit Decision", type="primary", use_container_width=True):
                feedback = {
                    "decision": decision,
                    "comments": comments,
                    "timestamp": datetime.now().isoformat(),
                    "user": "supply_planner"
                }
                process_human_decision(feedback)
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🔄 Decision panel will appear when human review is required.")
    
    # Agent Log Section (full width)
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.subheader("🤖 Agent Activity Log")
    render_enhanced_agent_log(state)


# =========================
# Main UI Components
# =========================
def render_hero_section():
    """Render the main hero section with status and controls"""
    state = st.session_state.workflow_state
    status_text, status_type, status_desc = get_status_info(getattr(state, "status", WorkflowStatus.IDLE))
    
    hero_html = f'''
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; color: #e2e8f0;">
                    AI Supply Chain Optimization
                </h1>
                <p style="margin: 8px 0 0; font-size: 1.1rem; color: #94a3b8;">
                    Autonomous agentic system for demand forecasting & inventory optimization
                </p>
            </div>
            <div class="status-badge">
                <span class="status-dot dot {status_type}"></span>
                <strong>{status_text}</strong>
            </div>
        </div>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">{status_desc}</p>
    </div>
    '''
    
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3, col4 = st.columns([2, 1, 1, 6])
    
    with col2:
        if st.button("Start Analysis", 
                     disabled=st.session_state.workflow_started or bool(config.validate_config()),
                     use_container_width=True):
            start_workflow()
            st.rerun()
    
    with col3:
        if st.button("Reset", use_container_width=True):
            reset_workflow()
            st.rerun()

def render_metrics_dashboard():
    """Render key metrics dashboard"""
    state = st.session_state.workflow_state
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Data Quality</h4>
            <div class="value">{getattr(state,"data_quality_score",0.0):.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Forecast Confidence</h4>
            <div class="value">{getattr(state,"forecast_confidence",0.0):.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Optimization Score</h4>
            <div class="value">{getattr(state,"optimization_confidence",0.0):.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        risk_score = getattr(state,"overall_risk_score",0.0)
        risk_level = "LOW" if risk_score < 0.3 else "MED" if risk_score < 0.7 else "HIGH"
        st.markdown(f'''
        <div class="metric-card">
            <h4>Risk Level</h4>
            <div class="value" style="color: {'#22c55e' if risk_score < 0.3 else '#f59e0b' if risk_score < 0.7 else '#ef4444'}">{risk_level}</div>
        </div>
        ''', unsafe_allow_html=True)

def render_workflow_content():
    """Render main workflow content based on current state"""
    state = st.session_state.workflow_state
    status = getattr(state, "status", WorkflowStatus.IDLE)
    
    if status == WorkflowStatus.IDLE:
        st.subheader("System Overview")
        st.markdown("""
        This AI-powered system orchestrates multiple specialized agents to optimize your supply chain:
        
        **🔍 Data Analyst Agent** - Validates data quality and identifies patterns  
        **📈 Demand Forecaster Agent** - Uses Prophet models for accurate demand prediction  
        **📦 Inventory Optimizer Agent** - Calculates optimal reorder points and safety stock  
        **⚠️ Risk Assessor Agent** - Evaluates supply chain risks and mitigation strategies  
        **✅ Validator Agent** - Ensures recommendations meet business constraints
        """)
        
        render_data_preview()
        
    elif status == WorkflowStatus.ERROR:
        st.error("System Error")
        st.markdown("The workflow encountered errors and could not complete.")
        if getattr(state, "errors", None):
            st.subheader("Error Details")
            for error in state.errors:
                st.error(f"• {error}")
        if getattr(state, "warnings", None):
            st.subheader("Warnings")
            for warning in state.warnings:
                st.warning(f"• {warning}")
                
    elif status in [WorkflowStatus.APPROVED, WorkflowStatus.REJECTED]:
        render_final_results(state, status)
        
    elif getattr(state, "requires_human_review", False):
        # This content will be in the AI Analysis tab
        st.info("Human review is required. Please check the AI Analysis tab for decision panel and detailed recommendations.")
        
    else:
        # Workflow in progress
        render_workflow_progress(state)

def render_workflow_progress(state):
    """Show workflow progress"""
    current_step = getattr(state, "current_step", "processing")
    progress = getattr(state, "progress", 0)
    status = getattr(state, "status", WorkflowStatus.IDLE)
    
    st.subheader(f"Processing: {current_step.replace('_', ' ').title()}")
    
    progress_messages = {
        "data_analysis": "Analyzing data quality and completeness patterns...",
        "forecasting": "Running AI-powered demand forecasting with Prophet models...",
        "optimizing": "Calculating optimal inventory parameters and safety stock levels...",
        "risk_assessment": "Evaluating supply chain risks and generating mitigation strategies...",
        "validation": "Validating recommendations against business constraints...",
        "generating_recommendations": "Generating comprehensive AI analysis and recommendations..."
    }
    
    message = progress_messages.get(current_step, "Processing supply chain optimization...")
    st.info(message)
    
    # Progress bar
    if progress > 0:
        st.progress(progress / 100)
    else:
        st.progress(0.7)  # Default progress for active processing
    
    # Auto-refresh while processing
    if status not in [WorkflowStatus.HUMAN_REVIEW, WorkflowStatus.APPROVED, 
                      WorkflowStatus.REJECTED, WorkflowStatus.ERROR]:
        st.rerun()

def render_final_results(state, status):
    """Render final results after approval/rejection"""
    if status == WorkflowStatus.APPROVED:
        st.success("✅ Recommendations Approved!")
        st.balloons()
        
        st.markdown("""
        ### Implementation Ready
        Your supply chain optimization recommendations have been approved and are ready for implementation.
        
        **Next Steps:**
        1. Download the optimization results
        2. Update your inventory management system  
        3. Monitor performance against forecasts
        4. Schedule regular reviews
        """)
        
    elif status == WorkflowStatus.REJECTED:
        st.error("❌ Recommendations Rejected")
        st.markdown("""
        ### Next Steps
        The recommendations were rejected. You can:
        1. Review the feedback provided
        2. Adjust input parameters
        3. Start a new workflow
        """)
    
    # Download button for results
    results_data = workflow_engine.get_results_data()
    if 'inventory_optimization' in results_data:
        df = results_data['inventory_optimization']
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"supply_chain_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Decision details
    if getattr(state, "human_feedback", None):
        with st.expander("📋 Decision Details"):
            st.json(state.human_feedback)


# =========================
# Tab Content Renderers
# =========================
def render_results_overview():
    """Render results overview tab"""
    st.subheader("Results Summary")
    
    results_data = workflow_engine.get_results_data()
    
    if 'inventory_optimization' in results_data:
        df = results_data['inventory_optimization']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total SKUs", df['SKU'].nunique() if 'SKU' in df.columns else 0)
        with col2:
            avg_demand = df['forecasted_demand'].mean() if 'forecasted_demand' in df.columns else 0
            st.metric("Avg Monthly Demand", f"{avg_demand:.0f}")
        with col3:
            avg_rop = df['reorder_point'].mean() if 'reorder_point' in df.columns else 0
            st.metric("Avg Reorder Point", f"{avg_rop:.0f}")
        with col4:
            avg_safety = df['safety_stock'].mean() if 'safety_stock' in df.columns else 0
            st.metric("Avg Safety Stock", f"{avg_safety:.0f}")
        
        # Data table
        st.subheader("Optimization Results")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("No optimization results available yet. Please run the workflow first.")

def render_demand_forecast_tab():
    """Render demand forecast tab"""
    st.subheader("Demand Forecast Analysis")
    
    results_data = workflow_engine.get_results_data()
    
    if 'demand_forecast' in results_data:
        df = results_data['demand_forecast']
        
        # Forecast accuracy metrics
        if 'MAE' in df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average MAE", f"{df['MAE'].mean():.2f}")
            with col2:
                st.metric("Average RMSE", f"{df['RMSE'].mean():.2f}")
            with col3:
                confidence = 1.0 - (df['MAE'].mean() / df['forecasted_demand'].mean()) if df['forecasted_demand'].mean() > 0 else 0
                st.metric("Forecast Confidence", f"{confidence:.1%}")
        
        # Forecast visualization
        if len(df) > 0:
            fig = px.line(df, x='Month', y='forecasted_demand', color='SKU',
                         title="3-Month Demand Forecast by SKU",
                         labels={'forecasted_demand': 'Forecasted Demand'})
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("Detailed Forecast Results")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("No demand forecast results available yet. Please run the workflow first.")

def render_inventory_optimization_tab():
    """Render inventory optimization tab"""
    st.subheader("Inventory Parameter Optimization")
    
    results_data = workflow_engine.get_results_data()
    
    if 'inventory_optimization' in results_data:
        df = results_data['inventory_optimization']
        
        # Filter controls
        if 'SKU' in df.columns:
            selected_skus = st.multiselect(
                "Filter by SKU:", 
                options=sorted(df['SKU'].unique()),
                default=[]
            )
            if selected_skus:
                df_filtered = df[df['SKU'].isin(selected_skus)]
            else:
                df_filtered = df
        else:
            df_filtered = df
        
        # Visualization
        if len(df_filtered) > 0 and 'reorder_point' in df_filtered.columns:
            fig = go.Figure()
            
            # Add bars for different metrics
            fig.add_trace(go.Bar(
                name='Reorder Point',
                x=df_filtered['SKU'],
                y=df_filtered['reorder_point'],
                marker_color='#8b5cf6'
            ))
            
            if 'safety_stock' in df_filtered.columns:
                fig.add_trace(go.Bar(
                    name='Safety Stock',
                    x=df_filtered['SKU'],
                    y=df_filtered['safety_stock'],
                    marker_color='#06b6d4'
                ))
            
            fig.update_layout(
                title='Inventory Parameters by SKU',
                xaxis_title='SKU',
                yaxis_title='Quantity',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Parameter statistics
        st.subheader("Parameter Statistics")
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df_filtered[numeric_cols].describe(), use_container_width=True)
        
        # Detailed results
        st.subheader("Detailed Parameters")
        st.dataframe(df_filtered, use_container_width=True)
        
    else:
        st.warning("No inventory optimization results available yet. Please run the workflow first.")


# =========================
# Main Application
# =========================
def main():
    # Hero section with status and controls
    render_hero_section()
    
    # Process steps visualization
    render_process_steps()
    
    # Key metrics dashboard
    render_metrics_dashboard()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 Demand Forecast", 
        "📦 Inventory Optimization", 
        "🤖 AI Analysis"
    ])
    
    with tab1:
        render_workflow_content()
    
    with tab2:
        render_demand_forecast_tab()
    
    with tab3:
        render_inventory_optimization_tab()
    
    with tab4:
        render_ai_analysis_with_decision()


if __name__ == "__main__":
    main()