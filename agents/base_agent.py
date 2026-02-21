# Create a new file: agents/base_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
from typing import Dict, Any, Optional
from config import SupplyChainConfig

logger = logging.getLogger(__name__)

class BaseSupplyChainAgent:
    """Base class for all supply chain agents with robust error handling"""
    
    def __init__(self, config: SupplyChainConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Optional[ChatGroq]:
        """Initialize LLM with proper error handling"""
        try:
            if not self.config.GROQ_API_KEY:
                logger.warning(f"{self.agent_name}: No GROQ API key found")
                return None
            
            return ChatGroq(
                groq_api_key=self.config.GROQ_API_KEY,
                model_name=self.config.GROQ_MODEL,
                temperature=0.1,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to initialize LLM: {e}")
            return None
    
    def _safe_llm_call(self, prompt: ChatPromptTemplate, **kwargs) -> Optional[str]:
        """Make LLM call with proper error handling and retries"""
        if not self.llm:
            logger.warning(f"{self.agent_name}: LLM not available, skipping AI analysis")
            return None
        
        try:
            messages = prompt.format_messages(**kwargs)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"{self.agent_name}: LLM call failed: {e}")
            return None
    
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
    
    def add_message_to_state(self, state: Dict[str, Any], content: str):
        """Add agent message to state safely"""
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append(
            AIMessage(content=f"{self.agent_name}: {content}")
        )