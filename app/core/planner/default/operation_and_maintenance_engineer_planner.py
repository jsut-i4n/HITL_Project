import asyncio
from typing import Any, List

from agentuniverse.agent.action.knowledge.store.document import Document
from agentuniverse.agent.action.knowledge.store.query import Query

from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.knowledge_manager import KnowledgeManager
from agentuniverse.agent.action.tool.tool_manager import ToolManager
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


class OperationAndMaintenanceEngineerAgentPlanner(Planner):
    """
    运维工程师智能体的工作流程.
    形成当前网络状态，用户状态，资源使用情况并进行分析产生报告
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
        self.run_all_actions(agent_model, planner_input, input_object)
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
        print("tool test", planner_input)
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
            print("operation_and_maintenance_engineer_planner_input:", planner_input)
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

    def run_all_actions(self, agent_model: AgentModel, planner_input: dict, input_object: InputObject):
        """Tool and knowledge processing.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
            input_object (InputObject): Agent input object.
        """
        action: dict = agent_model.action or dict()
        tools: list = action.get('tool') or list()
        knowledge: list = action.get('knowledge') or list()

        action_result: list = list()

        for tool_name in tools:
            tool = ToolManager().get_instance_obj(tool_name)
            if tool is None:
                continue
            tool_input = {key: input_object.get_data(key) for key in tool.input_keys}
            action_result.append(tool.run(**tool_input))



        for knowledge_name in knowledge:
            knowledge: Knowledge = KnowledgeManager().get_instance_obj(knowledge_name)
            if knowledge is None:
                continue
            knowledge_res: List[Document] = knowledge.store.query(
                Query(query_str=input_object.get_data(self.input_key), similarity_top_k=2), **input_object.to_dict())
            for document in knowledge_res:
                action_result.append(document.text)

        if planner_input['background']:
            planner_input['background'] += "\n"
        planner_input['background'] += "\n".join(action_result)
