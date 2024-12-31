from agentuniverse.agent.memory.chat_memory import ChatMemory
from agentuniverse.llm.default.qwen_openai_style_llm import QWenOpenAIStyleLLM


class HITLAgentMemory(ChatMemory):
    """The aU demo memory module."""

    def __init__(self, **kwargs):
        """The __init__ method.

        Some parameters, such as name/description/type/memory_key,
        are injected into this class by the demo.yaml configuration.


        Args:
            llm (LLM): the LLM instance used by this memory.
        """
        super().__init__(**kwargs)
