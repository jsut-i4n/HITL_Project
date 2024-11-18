from langchain.output_parsers.json import parse_json_markdown

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject


class SystemArchitectAgent(Agent):
    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['input']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['system_architect_result']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """
        agent_input['input'] = input_object.get_data('input')
        # 一定有客户经理的结果
        for framework in input_object.get_data('client_manager_result').get_data('framework'):
            agent_input['input'] += '\n' + framework
            input_object.add_data('sssss_framework', '\n' + framework)
            # TODO 查看input_object
        # 系统架构师获取运维工程师结果（不一定有）
        operation_and_maintenance_results = input_object.get_data('operation_and_maintenance_engineer_result')
        if operation_and_maintenance_results:
            try:
                operation_and_maintenance_engineer_result = operation_and_maintenance_results.get_data(
                    'operation_and_maintenance_engineer_result', [])
                print("xxx",operation_and_maintenance_engineer_result)
                agent_input['input'] += '\n' + operation_and_maintenance_engineer_result["suggestion"]
            except AttributeError:
                pass

        # 系统架构师获取调度员的结果（不一定有）
        dispatcher_agent_results = input_object.get_data('dispatcher_agent_result')
        if dispatcher_agent_results:
            try:
                dispatcher_agent_result = dispatcher_agent_results.get_data('dispatcher_agent_result', [])
                
                agent_input['input'] += '\n' + dispatcher_agent_result["output"]
            except AttributeError:
                pass

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
        print("rag", planner_result)
        return {"system_architect_result": planner_result}
