
from langchain.agents import create_agent, AgentState
from langgraph.store.memory import InMemoryStore
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver

class CustomAgentState(AgentState):
    run_id: str
    env_schema: dict
    episode_index: int
    step_index: int
    constants: list
    episodes: list


system_instructions = """You are an RL Agent operating inside an unknown python gym environment."""

class LLMAgent():
    def __init__(self, model="openai:gpt-5.1", session_id=None) -> None:
        if session_id is None:
            raise ValueError('session_id not defined')
        self.session_id = session_id
        self.agent = create_deep_agent(
            model=model,
            system_prompt=system_instructions,
            backend=self._make_backend,
            store=InMemoryStore(),
            checkpointer=MemorySaver(),
            subagents=[],
            tools=[],
        )
        
    def _make_backend(self, runtime):
        return CompositeBackend(
            default=StateBackend(runtime),  # Ephemeral storage
            routes={
                "/memories/": FilesystemBackend(root_dir=f"./runs/{self.session_id}/exports/", virtual_mode=True),  # Persistent storage
            }
        )

    def _invoke(self, prompt):
        """
        
        """
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": prompt}],
            "session_id": self.session_id,
        },
        {
            "configurable": {"thread_id": str(self.session_id)}
        })
        print(result["messages"][-1].content)
        return result["messages"][-1].content