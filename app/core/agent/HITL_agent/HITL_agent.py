from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject


class HITLAgent(Agent):
    def input_keys(self) -> list[str]:
        return ['input']

    def output_keys(self) -> list[str]:
        return ['output']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        agent_input['input'] = input_object.get_data('input')
        return agent_input

    def parse_result(self, planner_result: dict) -> dict:
        """
        HITL agent的planner输出格式如下,由HITL_planner.py文件决定
        {"result":
            [
                {
                "client_manager_result":example_content,
                "product_manager_result":example_content,
                "system_architect_result":example_content,
                "operation_and_maintenance_engineer_result":example_content,
                "dispatcher_agent_result":example_content
                }
            ]
        
        }

        return 的格式
        {
            "client_manager_result":example_content,
            "product_manager_result":example_content,
            "system_architect_result":example_content,
            "operation_and_maintenance_engineer_result":example_content,
            "dispatcher_agent_result":example_content
        }
        example_content的具体的内容需要使用.get_data().get(output)方法获取
        """
        return {"output": planner_result.get("result")[-1]}  # 获取最新的结果
