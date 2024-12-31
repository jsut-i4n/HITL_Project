# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/4/16 14:42
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: prompt_util.py
from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

from agentuniverse.agent.memory.enum import ChatMessageEnum
from agentuniverse.agent.memory.message import Message
from agentuniverse.llm.llm import LLM
from agentuniverse.llm.llm_manager import LLMManager
from agentuniverse.prompt.prompt_manager import PromptManager
from agentuniverse.prompt.prompt_model import AgentPromptModel
from agentuniverse.prompt.enum import PromptProcessEnum
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba.analyse
from snownlp import SnowNLP
import pandas as pd
import numpy as np

#关键词抽取
def keywords_extraction(text):
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
            # allow_speech_tags   --词性列表，用于过滤某些词性的词
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                                         pagerank_config={'alpha': 0.85, })
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    return keywords

def summarize_by_stuff(texts: List[str], llm: LLM, summary_prompt):
    """
    stuff summarization -- general method
    """
    stuff_chain = load_summarize_chain(llm.as_langchain(), chain_type='stuff', verbose=True,
                                       prompt=summary_prompt.as_langchain())
    return stuff_chain.run([Document(page_content=text) for text in texts])


def summarize_by_map_reduce(texts: List[str], llm: LLM, summary_prompt, combine_prompt):
    """
    map reduce summarization -- general method
    """
    map_reduce_chain = load_summarize_chain(llm.as_langchain(), chain_type='map_reduce', verbose=True,
                                            map_prompt=summary_prompt.as_langchain(),
                                            combine_prompt=combine_prompt.as_langchain())
    return map_reduce_chain.run([Document(page_content=text) for text in texts])


def split_text_on_tokens(text: str, text_token: int, chunk_size=800, chunk_overlap=100) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    # calculate the number of characters represented by each token.
    char_per_token = len(text) / text_token
    chunk_char_size = int(chunk_size * char_per_token)
    chunk_char_overlap = int(chunk_overlap * char_per_token)

    result = []
    current_position = 0

    while current_position + chunk_char_overlap < len(text):
        if current_position + chunk_char_size >= len(text):
            chunk = text[current_position:]
        else:
            chunk = text[current_position:current_position + chunk_char_size]

        result.append(chunk)
        current_position += chunk_char_size - chunk_char_overlap

    if len(result) == 0:
        result.append(text[current_position:])

    return result


def split_texts(texts: list[str], llm: LLM, chunk_size=800, chunk_overlap=100, retry=True) -> list[str]:
    """
    split texts into chunks with the fixed token length -- general method
    """
    try:
        split_texts_res = []
        for text in texts:
            text_token = llm.get_num_tokens(text)
            split_texts_res.extend(
                split_text_on_tokens(text=text, text_token=text_token, chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap))
        return split_texts_res
    except Exception as e:
        if retry:
            return split_texts(texts=texts, llm=llm, retry=False)
        raise ValueError("split text failed, exception=" + str(e))


def truncate_content(content: str, token_length: int, llm: LLM) -> str:
    """
    truncate the content based on the llm token limit
    """
    return str(split_texts(texts=[content], chunk_size=token_length, llm=llm)[0])


def generate_template(agent_prompt_model: AgentPromptModel, prompt_assemble_order: list[str]) -> str:
    """Convert the agent prompt model to an ordered list.

    Args:
        agent_prompt_model (AgentPromptModel): The agent prompt model.
        prompt_assemble_order (list[str]): The prompt assemble ordered list.
    Returns:
        list: The ordered list.
    """
    values = []
    for attr in prompt_assemble_order:
        value = getattr(agent_prompt_model, attr, None)
        if value is not None:
            values.append(value)

    return "\n".join(values)


def generate_chat_template(agent_prompt_model: AgentPromptModel, prompt_assemble_order: list[str]) -> list[Message]:
    """Convert the agent prompt model to the agentUniverse message list.

    Args:
        agent_prompt_model (AgentPromptModel): The agent prompt model.
        prompt_assemble_order (list[str]): The prompt assemble ordered list.
    Returns:
        list: The agentUniverse message list.
    """
    message_list = []
    for attr in prompt_assemble_order:
        value = getattr(agent_prompt_model, attr, None)
        if value is not None:
            message_list.append(
                Message(type=agent_prompt_model.get_message_type(attr), content=value))
    if message_list:
        # Integrate the system messages and put them in the first of the message list.
        system_messages = '\n'.join(msg.content for msg in message_list if msg.type == ChatMessageEnum.SYSTEM.value)
        if system_messages:
            message_list = list(filter(lambda msg: msg.type != ChatMessageEnum.SYSTEM.value, message_list))
            message_list.insert(0, Message(type=ChatMessageEnum.SYSTEM.value, content=system_messages))
    return message_list


def process_llm_token(agent_llm: LLM, lc_prompt_template, profile: dict, planner_input: dict):
    """Process the prompt template based on the prompt processor.

    Args:
        agent_llm (LLM): The agent llm.
        lc_prompt_template: The langchain prompt template.
        profile (dict): The profile.
        planner_input (dict): The planner input.
    """
    llm_model: dict = profile.get('llm_model')

    # get the prompt processor configuration
    prompt_processor: dict = llm_model.get('prompt_processor') or dict()
    prompt_processor_type: str = prompt_processor.get('type') or PromptProcessEnum.TRUNCATE.value
    prompt_processor_llm: str = prompt_processor.get('llm')

    # get the summary and combine prompt versions
    summary_prompt_version: str = prompt_processor.get('summary_prompt_version') or 'prompt_processor.summary_cn'
    combine_prompt_version: str = prompt_processor.get('combine_prompt_version') or 'prompt_processor.combine_cn'

    prompt_input_dict = {key: planner_input[key] for key in lc_prompt_template.input_variables if key in planner_input}

    # get the llm instance for prompt compression
    prompt_llm: LLM = LLMManager().get_instance_obj(prompt_processor_llm)

    if prompt_llm is None:
        prompt_llm = agent_llm

    prompt = lc_prompt_template.format(**prompt_input_dict)
    # get the number of tokens in the prompt
    prompt_tokens: int = agent_llm.get_num_tokens(prompt)

    input_tokens = agent_llm.max_context_length() - agent_llm.max_tokens
    if input_tokens <= 0:
        raise Exception("The current output max tokens limit is greater than the context length of the LLM model, "
                        "please adjust it by editing the `max_tokens` parameter in the llm yaml.")

    if prompt_tokens <= input_tokens:
        return

    process_prompt_type_enum = PromptProcessEnum.from_value(prompt_processor_type)

    # compress the background in the prompt
    content = planner_input.get('background')
    #1.关键词提取，从background中提取关键词
    keywords = keywords_extraction(content)
    output_array = [keyword.word for keyword in keywords]
    print(output_array)
    #print(keywords)

    #2.压缩方法的选择，如果背景长度小于200个单词，则使用映射-归约，否则使用填充
    if content:
        content_length = len(content.split())  # 计算单词数以判断长度
        if content_length < 200:  # 假设200个单词以下为较短且信息密集的文本
            # 使用映射-归约
            compressed_content = summarize_by_map_reduce(texts=split_texts([content], agent_llm),
                                                                  llm=prompt_llm,
                                                                  summary_prompt=PromptManager().get_instance_obj(
                                                                      summary_prompt_version),
                                                                  combine_prompt=PromptManager().get_instance_obj(
                                                                      combine_prompt_version))
        else:
            # 使用填充
            compressed_content = summarize_by_stuff(texts=[content], llm=prompt_llm,
                                                             summary_prompt=PromptManager().get_instance_obj(
                                                                 summary_prompt_version))
    # 3. 关键词的检测：检查压缩后的内容是否包括所有的关键词
    missing_keywords = [keyword for keyword in output_array if keyword not in compressed_content]

    if missing_keywords:
    # 如果有缺失的关键词，将它们添加到内容的后面
        compressed_content += ' ' + ' '.join(missing_keywords)
        print(f"Missing keywords added: {missing_keywords}")

    # 更新 planner_input 中的背景内容
    planner_input['background'] = compressed_content

    # 4. 再次判断，如果仍超出令牌限制，则使用截断
    prompt_tokens_after_addition = agent_llm.get_num_tokens(planner_input['background'])

    if prompt_tokens_after_addition > input_tokens:
    # 如果内容仍然超出令牌限制，进行截断
        planner_input['background'] = truncate_content(planner_input['background'], input_tokens, agent_llm)
        print("Content truncated due to token limit.")


    #if content:
        #if process_prompt_type_enum == PromptProcessEnum.TRUNCATE:
            #planner_input['background'] = truncate_content(content, input_tokens, agent_llm)
        #elif process_prompt_type_enum == PromptProcessEnum.STUFF:
            #planner_input['background'] = summarize_by_stuff(texts=[content], llm=prompt_llm,
                                                             #summary_prompt=PromptManager().get_instance_obj(
                                                                 #summary_prompt_version))
        #elif process_prompt_type_enum == PromptProcessEnum.MAP_REDUCE:
            #planner_input['background'] = summarize_by_map_reduce(texts=split_texts([content], agent_llm),
                                                                  #llm=prompt_llm,
                                                                  #summary_prompt=PromptManager().get_instance_obj(
                                                                      #summary_prompt_version),
                                                                  #combine_prompt=PromptManager().get_instance_obj(
                                                                      #combine_prompt_version))
