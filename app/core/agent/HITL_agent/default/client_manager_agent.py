import json

from langchain.output_parsers.json import parse_json_markdown

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject


class ClientManagerAgent(Agent):
    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['input']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['output']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """
        agent_input['input'] = input_object.get_data('input')  # 获取用户输入的文本
        agent_input['expert_framework'] = input_object.get_data('expert_framework')
        self.agent_model.profile.setdefault('prompt_version', 'default_planning_agent.cn')
        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """Planner result parser.

        Args:
            planner_result(dict): Planner result
        Returns:
            dict: Agent result object.
        """
        # 这里的输出结构是由client_manager_agent的提示词所定义的  -> client_manager_agent_cn.yaml
        output = planner_result.get('output')
        print("client_manager_agent output:", output)
        if isinstance(output, str):
            try:
                output = parse_json_markdown(output)
                planner_result['framework'] = output['framework']
                planner_result['thought'] = output['thought']
            except json.JSONDecodeError:
                # 如果output不是有效的JSON，返回原始的planner_result
                # 这里只是简单地返回了原始的planner_result
                return planner_result
        else:
            # 如果output既不是字符串也不是字典，
            # 返回了原始的planner_result
            return planner_result

            # 返回修改后的planner_result
        return planner_result
