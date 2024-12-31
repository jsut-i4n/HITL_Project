from langchain.output_parsers.json import parse_json_markdown

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject


class DispatcherAgent(Agent):
    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['input']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['dispatcher_agent_result']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """

        agent_input['input'] = input_object.get_data('input')
        agent_input['background'] = self.build_background(input_object)

        operation_and_maintenance_result = input_object.get_data('operation_and_maintenance_engineer_result')
        if operation_and_maintenance_result:
            try:
                operation_and_maintenance_engineer_result = operation_and_maintenance_result.get_data(
                    'operation_and_maintenance_engineer_result', [])
                
                agent_input["input"] = (operation_and_maintenance_engineer_result["suggestion"])
                
            except AttributeError:
                pass

        print("dispatcher_agent_input", agent_input)
        self.agent_model.profile.setdefault('prompt_version', 'default_expressing_agent.cn')
        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """Planner result parser.

        Args:
            planner_result(dict): Planner result
        Returns:
            dict: Agent result object.
        """
        return {'dispatcher_agent_result': planner_result}

    def build_background(self, input_object: InputObject) -> str:
        """Build the background knowledge.

        Args:
            input_object(InputObject): agent parameter object
        Returns:
            str: Background knowledge.
        """
        # 一定有系统架构师的结果
        dispatcher_agent_results = input_object.get_data('system_architect_result').get_data('system_architect_result', [])
        dispatcher_agent_results = [dispatcher_agent_results]
        operation_and_maintenance_result = input_object.get_data('operation_and_maintenance_engineer_result')
        
        # 获取运维工程师的结果（不一定）
        if operation_and_maintenance_result:
            try:
                operation_and_maintenance_engineer_result = operation_and_maintenance_result.get_data(
                    'operation_and_maintenance_engineer_result', [])
                
                dispatcher_agent_results.append(operation_and_maintenance_engineer_result["suggestion"])
                
                
            except AttributeError:
                pass

        knowledge_list = []
        for dispatcher_agent_result in dispatcher_agent_results:
            if isinstance(dispatcher_agent_result,
                          dict) and 'input' in dispatcher_agent_result and 'output' in dispatcher_agent_result:
                knowledge_list.append("question:" + dispatcher_agent_result.get('input'))
                knowledge_list.append("answer:" + dispatcher_agent_result.get('output'))

        return '\n\n'.join(knowledge_list)
