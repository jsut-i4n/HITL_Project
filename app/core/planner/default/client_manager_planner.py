import asyncio
import logging
from typing import Any, Optional, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory

from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.chat_memory import ChatMemory
from ..planner_base import Planner
# from agentuniverse.agent.plan.planner.planner import Planner
from agentuniverse.base.util.memory_util import generate_memories
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt
from agentuniverse.prompt.prompt_manager import PromptManager
from agentuniverse.prompt.prompt_model import AgentPromptModel

from ...scene_config.common_processor import CommonProcessor
from ...scene_config.data_format import extract_float, extract_continuous_digits

logging.getLogger().setLevel(logging.ERROR)


class ClientManagerPlanner(Planner):
    """
    继承planner_base.py中的Planner类，用于加载场景文件.
    客户经理智能体的工作流程.
    通过与人的交互，获取整个系统部署的基本信息，并根据这些信息获取社会，地区，组织，业务等背景知识
    """
    scene_templates: Optional[Dict] = None
    current_purpose: str = ''
    processors: Dict[str, Any] = {}

    slot_llm: ChatOllama = ChatOllama(model="qwen2:7b")

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
        res, slot_full = self.invoke_chain(agent_model, chain_with_history, planner_input, chat_history, input_object)

        if slot_full:
            # 槽位填充完毕，使用hitl agent 执行
            # 只有补充完整的信息才能执行client manager agent
            session_id = input_object.get_data('session_id')
            # 使用res填充planner_input的background
            planner_input['background'] += res
            print("planner_input: ", planner_input)
            res = chain_with_history.invoke(input=planner_input, config={"configurable": {"session_id": session_id}})
            return {**planner_input, self.output_key: res, 'chat_history': generate_memories(chat_history), 'slot_full': slot_full}
        return {**planner_input, self.output_key: res, 'chat_history': generate_memories(chat_history), 'slot_full': slot_full}

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

        if not input_object.get_data('output_stream'):

            # 在处理planner时，先判断用户的的意图与当前场景是否相关
            # 第一次是不会判断用户意图和场景是否相关,而实直接识别意图
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
            return '未命中场景', False

            # res = chain.invoke(input=planner_input, config={"configurable": {"session_id": "unused"}})
            # return res
        result = []
        for token in chain.stream(input=planner_input, config={"configurable": {"session_id": "unused"}}):
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


    def is_related_to_last_intent(self, user_input):
        """
        判断当前输入是否与上一次意图场景相关
        """
        # 意图相关性判断阈值0-1
        # TODO 后期可以放到config文件中 参数越高，场景匹配越严格
        RELATED_INTENT_THRESHOLD = 0.6

        if not self.current_purpose:
            return False
        scene_description = self.scene_templates[self.current_purpose]['description']
        prompt_template = """
        你是一个专业的场景判别师
        你需要判别用户输入与当前对话场景是否相关
        如果当前用户输入【{user_input}】提及电池状态或电池电量，场景就切换了相关系数为0.0，如果只是对能耗、时延等参数进行描述，一般可以认为场景没有切换。
        当前对话场景: {scene_description}
        这两次场景是否关联？（仅用小数回答关联度，说明：得分范围0.0至1.0，如果越相关，输出的值越靠近1.0。）
        """
        prompt = PromptTemplate(
            input_variables=["user_input", "scene_description"],
            template=prompt_template
        )
        chain = prompt | self.slot_llm | StrOutputParser()
        print("scene_description:", scene_description)
        result = chain.invoke({"user_input": user_input["input"], "scene_description": scene_description})  # 相关系数
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
        # template_key 某个场景  比如weather_query
        # template_info 场景的参数信息

        for template_key, template_info in self.scene_templates.items():
            purpose_options[str(index)] = template_key
            purpose_description[str(index)] = template_info["description"]
            index += 1

        options_prompt = "\n".join([f"{key}. {value} - 请回复{key}" for key, value in purpose_description.items()])
        options_prompt += "\n0. 其他场景 - 请回复0"
        prompt_template = "下面多种场景，需要你根据用户输入进行判断，只答选项\n{options_prompt}\n用户输入：{user_input}\n请回复序号："

        prompt = PromptTemplate(
            input_variables=["options_prompt", "user_input"],
            template=prompt_template
        )
        chain = prompt | self.slot_llm | StrOutputParser()

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
