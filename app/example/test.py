from agentuniverse.agent.output_object import OutputObject
from agentuniverse.base.agentuniverse import AgentUniverse
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.agent_manager import AgentManager
"""
test.py文件用来测试功能是否正常
正常使用请使用 my_agent.py 文件

"""

AgentUniverse().start(config_path='../../config/config.toml')


def chat(question: str):
    """ Peer agents example.

    The peer agents in agentUniverse become a chatbot and can ask questions to get the answer.
    """
    instance: Agent = AgentManager().get_instance_obj('HITL_agent')
    output_object: OutputObject = instance.run(input=question)

    # 获取每个agent的回答

    Client_Manager_Result = output_object.get_data("output").get("client_manager_result")
    client_manager_result: OutputObject = Client_Manager_Result.get_data("framework")

    Product_Manager_Result = output_object.get_data("output").get("product_manager_result")
    # 列表套字典[{'input':'content', 'output':'content'},{'input':'content', 'output':'content'},...]
    product_manager_result: OutputObject = Product_Manager_Result.get_data("product_manager_result")

    System_Architect_Result = output_object.get_data("output").get("system_architect_result")
    system_architect_result: OutputObject = System_Architect_Result.get_data("system_architect_result").get("output")

    Operation_And_Maintenance_Engineer_Result = output_object.get_data("output").get("operation_and_maintenance_engineer_result")
    operation_and_maintenance_engineer_result: OutputObject = Operation_And_Maintenance_Engineer_Result.get_data("operation_and_maintenance_engineer_result").get("output")

    Dispatcher_Agent_Result = output_object.get_data("output").get("dispatcher_agent_result")
    dispatcher_agent_result: OutputObject = Dispatcher_Agent_Result.get_data("dispatcher_agent_result").get("output")

    print(f"Client Manager: {client_manager_result}")
    print(f"Product Manager: {product_manager_result}")
    print(f"System Architect: {system_architect_result}")
    print(f"Operation and Maintenance Engineer: {operation_and_maintenance_engineer_result}")
    print(f"Dispatcher Agent: {dispatcher_agent_result}")


if __name__ == '__main__':
    chat("当前我的系统部署在中国北京")

