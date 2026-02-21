# utils/tooling.py
from datetime import datetime
from state import make_serializable

def tools_system_message(config) -> str:
    scripts = []
    if getattr(config, "PROPHET_SCRIPT", None):
        scripts.append(str(config.PROPHET_SCRIPT))
    if getattr(config, "REORDERPOINT_SCRIPT", None):
        scripts.append(str(config.REORDERPOINT_SCRIPT))
    return (
        "TOOLS:\n"
        "- read_csv_preview(path, n=5)\n"
        "- compute_stats(path, columns=None)\n"
        "- call_python_script(path)\n"
        f"Available scripts: {', '.join(scripts) or 'None discovered'}\n"
        "If you decide to use a tool, state it explicitly (one line JSON), e.g. "
        '{"action":"read_csv_preview","args":{"path":"...","n":5}} '
        "then continue reasoning with the result. If no tool needed, proceed."
    )

def log_agent_action(state, agent_name, action, args=None, status="ok", result_preview=None):
    try:
        entry = {
            "ts": datetime.now().isoformat(),
            "agent": agent_name,
            "action": action,
            "args": make_serializable(args or {}),
            "status": status,
            "result_preview": str(result_preview)[:600] if result_preview is not None else None,
        }
        state.setdefault("agent_actions", []).append(entry)
    except Exception:
        pass
