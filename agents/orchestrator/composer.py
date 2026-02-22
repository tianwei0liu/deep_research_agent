"""
Composer Node: Synthesizes final report from completed tasks.
"""

import logging
from deep_research_agent.agents.orchestrator.state import OrchestratorState, TaskStatus
from deep_research_agent.agents.orchestrator.prompts import OrchestratorPrompts
from deep_research_agent.config import Settings
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


class Composer:
    """
    Synthesizes the final answer from completed tasks using a generative model.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, state: OrchestratorState, config: RunnableConfig) -> dict:
        """
        Synthesizes the final answer from completed tasks.
        """
        messages = state.get("messages", [])
        todos = state.get("todos", [])
        
        # Extract user query from messages
        user_query = "Unknown Query"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
            elif hasattr(msg, "role") and msg.role == "user": # Fallback if we mix types
                 user_query = msg.content
                 break
        
        # Filter for completed tasks (or failed ones, to report what we couldn't find)
        # in Dynamic Supervisor, we might have tasks that are still pending but we decided to finish.
        # We should include all relevant info.
        relevant_tasks = [t for t in todos if t.status in [TaskStatus.COMPLETED, TaskStatus.PARTIAL, TaskStatus.FAILED]]
        
        if not relevant_tasks:
            self.logger.warning("Composer running with no completed tasks.")
            # We still proceed, maybe the supervisor has enough info in context?
            # But prompts expects tasks.
        
        settings = Settings.load()
        client = genai.Client(api_key=settings.require_gemini_api_key())
        model_id = settings.composer_model

        prompt = OrchestratorPrompts.build_composer_prompt(user_query, relevant_tasks)

        # Use native Gemini for composer as before
        try:
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=settings.composer_temperature,
                    thinking_config=types.ThinkingConfig(thinking_level=settings.composer_thinking_level)
                )
            )
            
            final_report = response.candidates[0].content.parts[0].text
            self.logger.info("Composer generated final report.")
            return {"final_report": final_report}

        except Exception as e:
            self.logger.error(f"Composer failed: {e}")
            return {"final_report": f"Error generating report: {e}"}
