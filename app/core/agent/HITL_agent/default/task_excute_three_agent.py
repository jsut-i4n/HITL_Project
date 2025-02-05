from langchain.output_parsers.json import parse_json_markdown

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject


class TaskExcuteThreeAgent(Agent):
    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['input']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['task_excute_three_agent_result']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """
        x = input_object.get_data('client_manager_result').get_data('framework')
        for item in x:
            for k, v in item.items():
                if k == '文化':
                    agent_input['input'] = v
        # agent_input['input'] = input_object.get_data('input')
        
        # for framework in input_object.get_data('client_manager_result').get_data('framework'):
        #     agent_input['input'] += framework[2]
        #     input_object.add_data('sssss_framework', '\n' + framework)
        # agent_input['input'] = input_object.get_data('client_manager_result').get_data('framework')[2]
            
        self.agent_model.profile.setdefault('prompt_version', 'default_planning_agent.cn')

        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """Planner result parser.

        Args:
            planner_result(dict): Planner result
        Returns:
            dict: Agent result object.
        """
        output_object = OutputObject(planner_result)
        planner_result['background'] = output_object.get_data('background').replace("\n", "")  # 获取文档内容
       
        return {"task_excute_three_agent_result": planner_result}
