import copy

from agentuniverse.agent.plan.planner.planner import Planner
from langchain.output_parsers.json import parse_json_markdown

from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor

from typing import Any, Optional

from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.plan.planner.planner_manager import PlannerManager

from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject


class ProductManagerAgent(Agent):
    # 多线程操作
    executor: Optional[Any] = ThreadPoolExecutor(max_workers=10, thread_name_prefix="product_manager_planner_agent")

    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['client_manager_result']  # 用于检查是否用客户经理执行的结果

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['product_manager_result']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        """Agent parameter parsing.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input preparsed by the agent.
        Returns:
            dict: agent input parsed from `input_object` by the user.
        """
        agent_input['input'] = input_object.get_data('input')  # 用户原始输入
        agent_input['framework'] = input_object.get_data('client_manager_result').get_data('framework')
        self.agent_model.profile.setdefault('prompt_version', 'default_executing_agent.cn')
        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """Planner result parser.

        Args:
            planner_result(dict): Planner result
        Returns:
            dict: Agent result object.
        """
        llm_result = []
        product_manager_result = []
        futures = planner_result.get('futures')
        for future in futures:
            task_result = future.result()
            llm_result.append(task_result)
            product_manager_result.append({
                'input': task_result['input'], 'output': task_result['output']
            })

        return {'product_manager_result': product_manager_result, 'llm_result': llm_result}

    def execute(self, input_object: InputObject, agent_input: dict) -> dict:
        """Execute agent instance.

        Args:
            input_object (InputObject): input parameters passed by the user.
            agent_input (dict): agent input parsed from `input_object` by the user.

        Returns:
            dict: Agent result object.
        """
        framework = agent_input.get('framework', [])
        futures = []
        for task in framework:
            agent_input_copy: dict = copy.deepcopy(agent_input)
            agent_input_copy['input'] = task
            planner: Planner = PlannerManager().get_instance_obj(self.agent_model.plan.get('planner').get('name'))
            futures.append(
                self.executor.submit(planner.invoke, self.agent_model, agent_input_copy,
                                     self.process_intput_object(input_object, task, planner.input_key)))
        wait(futures, return_when=ALL_COMPLETED)
        return {'futures': futures}

    def process_intput_object(self, input_object: InputObject, subtask: str, planner_input_key: str) -> InputObject:
        """Process input object for the executing agent.

        Args:
            input_object (InputObject): input parameters passed by the user.
            subtask (str): subtask to be executed.
            planner_input_key (str): planner input key.

        Returns:
            input_object (InputObject): processed input object.
        """
        # get agent toolsets.
        action: dict = self.agent_model.action or dict()
        tools: list = action.get('tool') or list()
        input_object_copy: InputObject = copy.deepcopy(input_object)
        # wrap input_object for agent knowledge.
        input_object_copy.add_data(planner_input_key, subtask)
        # wrap input_object for agent toolsets.
        for tool_name in tools:
            tool = ToolManager().get_instance_obj(tool_name)
            if tool is None:
                continue
            # note: only insert the first key of tool input.
            input_object_copy.add_data(tool.input_keys[0], subtask)
        return input_object_copy
