import logging

from agentuniverse.base.util.env_util import get_from_env
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from .import scene_prompts
from .helpers import get_raw_slot, get_dynamic_example, extract_json_from_string, update_slot, \
    is_slot_fully_filled, format_name_value_for_logging, get_slot_update_message, get_slot_query_user_message

from .scene_processor import SceneProcessor
from .chat_history import get_chat_history

class CommonProcessor(SceneProcessor):
    def __init__(self, scene_config):
        parameters = scene_config["parameters"]
        self.scene_config = scene_config
        self.scene_name = scene_config["name"]
        self.slot_template = get_raw_slot(parameters)
        self.slot_dynamic_example = get_dynamic_example(scene_config)
        self.slot = get_raw_slot(parameters)
        self.scene_prompts = scene_prompts
        self.llm = ChatOllama(model='qwen2:7b')

    def process(self, user_input, context, input_object):
        # 处理用户输入，更新槽位，检查完整性，以及与用户交互
        # 先检查本次用户输入是否有信息补充，保存补充后的结果   编写程序进行字符串value值diff对比，判断是否有更新
        # message是提示词模板
        # user_input 用户输入
        message = get_slot_update_message(self.scene_name, self.slot_dynamic_example, self.slot_template,
                                          user_input["input"])

        session_id = input_object.get_data('session_id')

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个擅长信息抽取的助手,只对输入内容进行信息抽取"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | self.llm
        user_input["get_slot_message"] = message
        #  此处是获取聊天历史，但是因为获取了聊天历史，会导致识意图切换时，错误识别到上个场景的信息，因此需要这部分的记忆功能
        # user_input["chat_history"] = get_chat_history(session_id, 3)
        # new_info_json_raw 代表抽取到的json格式的词槽
        print("user_input:", user_input)
        new_info_json_raw = self.llm.invoke(message)
        current_values = extract_json_from_string(new_info_json_raw.content)
        logging.debug('current_values: %s', current_values)
        logging.debug('slot update before: %s', self.slot)
        update_slot(current_values, self.slot)  # 更新槽位
        print("self.slot:", self.slot)
        logging.debug('slot update after: %s', self.slot)
        # 判断参数是否已经全部补全
        if is_slot_fully_filled(self.slot):
            return self.respond_with_complete_data()
        else:
            return self.ask_user_for_missing_data(user_input)  # user_input还有对话历史、时间等

    def respond_with_complete_data(self):
        slot_full = True  # 槽位是否已完全填充标志位
        # 当所有数据都准备好后的响应
        logging.debug(f'%s ------ 参数已完整，详细参数如下', self.scene_name)
        logging.debug(format_name_value_for_logging(self.slot))
        logging.debug(f'正在请求%sAPI，请稍后……', self.scene_name)
        return format_name_value_for_logging(self.slot), slot_full

    def ask_user_for_missing_data(self, user_input):
        slot_full = False  # 槽位是否已完全填充标志位
        message = get_slot_query_user_message(self.scene_name, self.slot, user_input["input"])
        print("message:", message)
        # 请求用户填写缺失的数据

        result = self.llm.invoke(message)

        return result.content, slot_full
