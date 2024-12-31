"""Base class for Planner."""
from abc import abstractmethod
import logging
from queue import Queue
from typing import Optional, List, Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.knowledge_manager import KnowledgeManager
from agentuniverse.agent.action.knowledge.store.document import Document
from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent_manager import AgentManager
from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.chat_memory import ChatMemory
from agentuniverse.agent.memory.memory import Memory
from agentuniverse.agent.memory.message import Message
from agentuniverse.agent.memory.memory_manager import MemoryManager
from agentuniverse.base.component.component_base import ComponentBase
from agentuniverse.base.component.component_enum import ComponentEnum
from agentuniverse.base.config.component_configer.configers.planner_configer import PlannerConfiger
from agentuniverse.llm.llm import LLM
from agentuniverse.llm.llm_manager import LLMManager
from agentuniverse.prompt.prompt import Prompt
from agentuniverse.base.util.memory_util import generate_messages

from langchain_community.chat_models import ChatOllama

from ..scene_config.common_processor import CommonProcessor
from ..scene_config.data_format import extract_continuous_digits, extract_float
from ..scene_config.helpers import load_all_scene_configs

logging.getLogger().setLevel(logging.ERROR)


class Planner(ComponentBase):
    """
    Base class for all planners.

    All planners should inherit from this class
    """
    name: Optional[str] = None
    description: Optional[str] = None
    output_key: str = 'output'
    input_key: str = 'input'
    prompt_assemble_order: list = ['introduction', 'target', 'instruction']

    def __init__(self, scene_templates: Optional[Dict] = None):
        """Initialize the ComponentBase."""
        super().__init__(component_type=ComponentEnum.PLANNER)

        # 词槽设置
        if scene_templates is None:
            self.scene_templates: Dict = load_all_scene_configs()
        self.current_purpose: str = ''
        self.processors = {}
        self.slot_llm = ChatOllama(model="qwen2:7b")

    @abstractmethod
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
        pass

    def handle_memory(self, agent_model: AgentModel, planner_input: dict) -> ChatMemory | None:
        """Memory module processing.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
        Returns:
             Memory: The memory.
        """
        chat_history: list = planner_input.get('chat_history')
        memory_name = agent_model.memory.get('name')
        memory: ChatMemory = MemoryManager().get_instance_obj(component_instance_name=memory_name)
        if memory is None:
            return None

        llm_model = agent_model.memory.get('llm_model') or dict()
        llm_name = llm_model.get('name') or agent_model.profile.get('llm_model').get('name')

        messages: list[Message] = generate_messages(chat_history)
        llm: LLM = LLMManager().get_instance_obj(llm_name)
        params: dict = dict()
        params['messages'] = messages
        params['llm'] = llm
        params['input_key'] = self.input_key
        params['output_key'] = self.output_key
        return memory.set_by_agent_model(**params)

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
        agents: list = action.get('agent') or list()

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
            knowledge_res: List[Document] = knowledge.query_knowledge(query_str=input_object.get_data(self.input_key),
                                                                      **input_object.to_dict())
            for document in knowledge_res:
                action_result.append(document.text)

        for agent_name in agents:
            agent = AgentManager().get_instance_obj(agent_name)
            if agent is None:
                continue
            agent_input = {key: input_object.get_data(key) for key in agent.input_keys()}
            output_object = agent.run(**agent_input)
            action_result.append("\n".join([output_object.get_data(key)
                                            for key in agent.output_keys()
                                            if output_object.get_data(key) is not None]))

        planner_input['background'] = planner_input['background'] or '' + "\n".join(action_result)

    def handle_prompt(self, agent_model: AgentModel, planner_input: dict):
        """Prompt module processing.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
        Returns:
            Prompt: The prompt instance.
        """
        pass

    def handle_llm(self, agent_model: AgentModel) -> LLM:
        """Language model module processing.

        Args:
            agent_model (AgentModel): Agent model object.
        Returns:
            LLM: The language model.
        """
        llm_name = agent_model.profile.get('llm_model').get('name')
        llm: LLM = LLMManager().get_instance_obj(component_instance_name=llm_name, new_instance=True)
        llm.set_by_agent_model(**agent_model.profile.get('llm_model'))
        return llm

    def initialize_by_component_configer(self, component_configer: PlannerConfiger) -> 'Planner':
        """Initialize the planner by the PlannerConfiger object.

        Args:
            component_configer(PlannerConfiger): the PlannerConfiger object
        Returns:
            Planner: the planner object
        """
        self.name = component_configer.name
        self.description = component_configer.description
        self.input_key = component_configer.input_key or self.input_key
        self.output_key = component_configer.output_key or self.output_key
        return self

    @staticmethod
    def stream_output(input_object: InputObject, data: dict):
        """Stream output.

        Args:
            input_object (InputObject): Agent input object.
            data (dict): The data to be streamed.
        """
        output_stream: Queue = input_object.get_data('output_stream', None)
        if output_stream is None:
            return
        output_stream.put_nowait(data)

    def invoke_chain(self, agent_model: AgentModel, chain: RunnableSerializable[Any, str], planner_input: dict,
                     chat_history,
                     input_object: InputObject):

        session_id = input_object.get_data('session_id')

        if session_id is None:
            session_id = "unprovided"
        if not input_object.get_data('output_stream'):

            # 在处理planner时，先判断用户的的意图与当前场景是否相关
            # 项目第一次是不会判断用户意图和场景是否相关
            if self.is_related_to_last_intent(planner_input):
                pass
            else:
                # 不相关时，重新识别意图
                self.recognize_intent(planner_input)
            logging.info('current_purpose: %s', self.current_purpose)
            if self.current_purpose in self.scene_templates:
                # 根据场景模板调用相应场景的处理逻辑
                self.get_processor_for_scene(self.current_purpose)
                # 调用抽象类process方法
                return self.processors[self.current_purpose].process(planner_input, None, input_object)
            return '未命中场景'

            # res = chain.invoke(input=planner_input, config={"configurable": {"session_id": session_id}})
            # return res
        result = []
        for token in chain.stream(input=planner_input, config={"configurable": {"session_id": session_id}}):

            self.stream_output(input_object, {
                'type': 'token',
                'data': {
                    'chunk': token,
                    'agent_info': agent_model.info
                }
            })
            result.append(token)
        return "".join(result)

        ####

    def is_related_to_last_intent(self, user_input):
        """
        判断当前输入是否与上一次意图场景相关
        """
        # 意图相关性判断阈值0-1
        # TODO 后期可以放到config文件中 越高，场景匹配越严格
        RELATED_INTENT_THRESHOLD = 0.5

        if not self.current_purpose:
            return False
        scene_description = self.scene_templates[self.current_purpose]['description']
        prompt_template = "判断当前用户输入内容与当前对话场景的关联性:\n\n当前对话场景: {scene_description}\n当前用户输入: {user_input}\n\n这两次输入是否关联（仅用小数回答关联度，得分范围0.0至1.0）"
        prompt = PromptTemplate(
            input_variables=["scene_description", "user_input"],
            template=prompt_template
        )
        chain = prompt | self.slot_llm | StrOutputParser()

        print("user_input", user_input)
        print("scene_description: ", scene_description)
        result = chain.invoke({"scene_description": scene_description, "user_input": user_input["input"]})
        return extract_float(result) > RELATED_INTENT_THRESHOLD

    @staticmethod
    def load_scene_processor(self, scene_config):
        try:
            return CommonProcessor(scene_config)
        except (ImportError, AttributeError, KeyError):
            raise ImportError(f"未找到场景处理器 scene_config: {scene_config}")

    def recognize_intent(self, user_input):
        # 根据场景模板生成选项
        purpose_options = {}
        purpose_description = {}
        index = 1
        print("self.scene_templates: ", self.scene_templates)
        # template_key 某个场景  比如weather_query
        # template_info 场景的参数信息

        for template_key, template_info in self.scene_templates.items():
            purpose_options[str(index)] = template_key
            purpose_description[str(index)] = template_info["description"]
            index += 1
        # 1. 天气信息查询服务 - 请回复1
        options_prompt = "\n".join([f"{key}. {value} - 请回复{key}" for key, value in purpose_description.items()])
        options_prompt += "\n0. 其他场景 - 请回复0"
        prompt_template = "有下面多种场景，需要你根据用户输入进行判断，只答选项\n{options_prompt}\n用户输入：{user_input}\n请回复序号："

        prompt = PromptTemplate(
            input_variables=["options_prompt", "user_input"],
            template=prompt_template
        )
        chain = prompt | self.slot_llm | StrOutputParser()

        # 发送选项给用户
        # print("user_input: ", user_input)
        # like user_input:  {'chat_history': '', 'background': '', 'date': '2024-09-10', 'input': '订机票'}

        user_choice = chain.invoke({"options_prompt": options_prompt, "user_input": user_input["input"]})

        logging.debug(f'purpose_options: %s', purpose_options)
        logging.debug(f'user_choice: %s', user_choice)

        # 为了防止大模型未按照要求输出，使用正则表达式，匹配数字
        user_choices = extract_continuous_digits(user_choice)
        print("user_choices: ", user_choices)
        # 根据用户选择获取对应场景
        if user_choices and user_choices[0] != '0':

            self.current_purpose = purpose_options[user_choices[0]]

        if self.current_purpose:
            print(f"用户选择了场景：{self.scene_templates[self.current_purpose]['name']}")
            # TODO: 这里可以继续处理其他逻辑
        else:
            # 用户输入的选项无效的情况，可以进行相应的处理
            print("无效的选项，请重新选择")

    def get_processor_for_scene(self, scene_name):
        if scene_name in self.processors:
            return self.processors[scene_name]

        scene_config = self.scene_templates.get(scene_name)
        if not scene_config:
            raise ValueError(f"未找到名为{scene_name}的场景配置")

        processor_class = self.load_scene_processor(self, scene_config)
        self.processors[scene_name] = processor_class
        return self.processors[scene_name]
