from typing import Any


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory

from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.chat_memory import ChatMemory
from agentuniverse.agent.plan.planner.planner import Planner
from agentuniverse.base.util.memory_util import generate_memories
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt
from agentuniverse.prompt.prompt_manager import PromptManager
from agentuniverse.prompt.prompt_model import AgentPromptModel


class ProductManagerPlanner(Planner):
    """
    产品经理智能体的工作流程.
    获取客户和业务需求，系统结构，工业互联网业务相关的需求，形成客户工作流程和相关详细描述和指标要求
    """

    def invoke(self, agent_model: AgentModel, planner_input: dict,
               input_object: InputObject) -> dict:
        """Invoke the planner.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
            input_object (InputObject): The input parameters passed by the user.
        Returns:
            dict: The planner result.
        """
        memory: ChatMemory = self.handle_memory(agent_model, planner_input)

        llm: LLM = self.handle_llm(agent_model)

        prompt: Prompt = self.handle_prompt(agent_model, planner_input)
        process_llm_token(llm, prompt.as_langchain(), agent_model.profile, planner_input)

        chat_history = memory.as_langchain().chat_memory if memory else InMemoryChatMessageHistory()

        chain_with_history = RunnableWithMessageHistory(
            prompt.as_langchain() | llm.as_langchain(),
            lambda session_id: chat_history,
            history_messages_key="chat_history",
            input_messages_key=self.input_key,
        ) | StrOutputParser()
        res = self.invoke_chain(agent_model, chain_with_history, planner_input, chat_history, input_object)
        return {**planner_input, self.output_key: res, 'chat_history': generate_memories(chat_history)}

    def handle_prompt(self, agent_model: AgentModel, planner_input: dict) -> Prompt:
        """Prompt module processing.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
        Returns:
            Prompt: The prompt instance.
        """
        expert_framework = planner_input.pop('expert_framework', '') or ''

        profile: dict = agent_model.profile

        profile_instruction = profile.get('instruction')
        profile_instruction = expert_framework + profile_instruction if profile_instruction else profile_instruction

        profile_prompt_model: AgentPromptModel = AgentPromptModel(introduction=profile.get('introduction'),
                                                                  target=profile.get('target'),
                                                                  instruction=profile_instruction)

        # get the prompt by the prompt version
        prompt_version: str = profile.get('prompt_version')
        version_prompt: Prompt = PromptManager().get_instance_obj(prompt_version)

        if version_prompt is None and not profile_prompt_model:
            raise Exception("Either the `prompt_version` or `introduction & target & instruction`"
                            " in agent profile configuration should be provided.")
        if version_prompt:
            version_prompt_model: AgentPromptModel = AgentPromptModel(
                introduction=getattr(version_prompt, 'introduction', ''),
                target=getattr(version_prompt, 'target', ''),
                instruction=expert_framework + getattr(version_prompt, 'instruction', ''))
            profile_prompt_model = profile_prompt_model + version_prompt_model

        return Prompt().build_prompt(profile_prompt_model, self.prompt_assemble_order)

    def invoke_chain(self, agent_model: AgentModel, chain: RunnableSerializable[Any, str], planner_input: dict,
                     chat_history,
                     input_object: InputObject):
        session_id = input_object.get_data('session_id')
        if not input_object.get_data('output_stream'):
            print("case1")
            print("product_manager_planner_input:", planner_input)
            res = chain.invoke(input=planner_input, config={"configurable": {"session_id": session_id}})
            return res
        result = []
        for token in chain.stream(input=planner_input, config={"configurable": {"session_id": session_id}}):
            print("case 2")
            self.stream_output(input_object, {
                'type': 'token',
                'data': {
                    'chunk': token,
                    'agent_info': agent_model.info
                }
            })
            result.append(token)
        return "".join(result)

