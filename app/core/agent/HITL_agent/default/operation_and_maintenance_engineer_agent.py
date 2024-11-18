from langchain.output_parsers.json import parse_json_markdown

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject


class OperationAndMaintenanceEngineerAgent(Agent):
    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['dispatcher_agent_result']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['operation_and_maintenance_engineer_result']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """
        agent_input['input'] = input_object.get_data('input')
        # agent_input['background'] = self.build_background(input_object)
        agent_input['dispatcher_agent_result'] = input_object.get_data('dispatcher_agent_result').get_data('dispatcher_agent_result').get('output')
        self.agent_model.profile.setdefault('prompt_version', 'default_planning_agent.cn')
        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """Planner result parser.

        Args:
            planner_result(dict): Planner result
        Returns:
            dict: Agent result object.
        """
        agent_result = dict()
        # TODO 后面看一下OAM的background
        output = planner_result.get('output')
        output = parse_json_markdown(output)
        is_useful = output.get('is_useful')
        if is_useful is None:
            is_useful = False
        is_useful = bool(is_useful)
        if is_useful:
            score = 80
        else:
            score = 0

        agent_result['output'] = output
        agent_result['score'] = score
        agent_result['suggestion'] = output.get('suggestion')

        return {"operation_and_maintenance_engineer_result": agent_result}

    def build_background(self, input_object: InputObject) -> str:
        """Build the background knowledge.

        Args:
            input_object(InputObject): agent parameter object
        Returns:
            str: Background knowledge.
        """
        # 获取调度员的结果
        dispatcher_agent_results = input_object.get_data('dispatcher_agent_result').get_data(
            'dispatcher_agent_result', [])
        dispatcher_agent_results = [dispatcher_agent_results]
        print()
        print("dispatcher_agent_results", dispatcher_agent_results)

        knowledge_list = []
        for dispatcher_agent_result in dispatcher_agent_results:
            if isinstance(dispatcher_agent_result,
                          dict) and 'input' in dispatcher_agent_result and 'output' in dispatcher_agent_result:
                knowledge_list.append("question:" + dispatcher_agent_result.get('input'))
                knowledge_list.append("answer:" + dispatcher_agent_result.get('output'))

        return '\n\n'.join(knowledge_list)
