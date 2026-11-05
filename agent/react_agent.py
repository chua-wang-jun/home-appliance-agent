from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from chat_history.memory import ChatMemory
from model.factlory import chat_model
from utils.prompt_loader import load_system_prompts

class ReactAgent:
    def __init__(self):
        self.memory = ChatMemory()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.memory.checkpointer,
        )

    def close(self):
        self.memory.close()

    @staticmethod
    def _build_runtime_context(context: dict | None = None) -> dict:
        runtime_context = {
            "report": False,
            "user_id": "",
            "period": "",
            "location": "",
            "weather": "",
            "usage_data": "",
            "knowledge_context": "",
        }
        if context:
            runtime_context.update(context)
        return runtime_context

    @staticmethod
    def _message_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def _thread_config(self, thread_id: str) -> dict:
        return self.memory.thread_config(thread_id)

    def execute_stream(
        self,
        query: str,
        context: dict | None = None,
        thread_id: str = "default",
    ):
        input_dict = {"messages": [{"role": "user", "content": query}]}
        runtime_context = self._build_runtime_context(context)
        config = self._thread_config(thread_id)

        for chunk in self.agent.stream(
            input_dict,
            config=config,
            stream_mode="values",
            context=runtime_context,
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    def execute_chat_stream(
        self,
        query: str,
        context: dict | None = None,
        thread_id: str = "default",
    ):
        input_dict = {"messages": [{"role": "user", "content": query}]}
        runtime_context = self._build_runtime_context(context)
        config = self._thread_config(thread_id)

        for message, _metadata in self.agent.stream(
            input_dict,
            config=config,
            stream_mode="messages",
            context=runtime_context,
        ):
            if not isinstance(message, (AIMessage, AIMessageChunk)):
                continue
            text = self._message_text(message.content)
            if text:
                yield text

    def get_thread_messages(self, thread_id: str, limit: int = 200) -> list[dict]:
        return self.memory.get_thread_messages(self.agent, thread_id, limit)

    def list_threads(self, checkpoint_limit: int = 500) -> list[dict]:
        return self.memory.list_threads(self.agent, checkpoint_limit)

    def delete_thread(self, thread_id: str) -> None:
        self.memory.delete_thread(thread_id)


if __name__ == "__main__":
    agent = ReactAgent()
    try:
        for chunk in agent.execute_stream("请生成本月家庭设备健康与能耗报告"):
            print(chunk, end="", flush=True)
    finally:
        agent.close()
