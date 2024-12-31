import streamlit as st
from PIL import Image
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.output_object import OutputObject
from agentuniverse.base.agentuniverse import AgentUniverse
from agentuniverse.agent.agent_manager import AgentManager

# 对话图标设置
avatar_icon = {
    "user": Image.open('assets/user.png'),
    "client_manager": Image.open('assets/client_manager.png'),
    'task_execute_one_agent':Image.open('assets/task_execute_one_agent.png'),
    'task_execute_two_agent':Image.open('assets/task_execute_two_agent.png'),
    'task_execute_three_agent':Image.open('assets/task_execute_three_agent.png'),
    "product_manager": Image.open('assets/product_manager.png'),
    "system_architect": Image.open('assets/system_architect.png'),
    "operation_and_maintenance_engineer_agent": Image.open('assets/operation_and_maintenance_engineer_agent.png'),
    "dispatcher_agent": Image.open('assets/dispatcher_agent.png'),
               }


agent_universe_initialized = False
hitl_agent_instance = None


def initialize_agent_universe_and_agent():
    """
    初始化  确保AgentUniverse的所有组件只被注册一次
    """
    global agent_universe_initialized, hitl_agent_instance
    if not agent_universe_initialized:
        AgentUniverse().start(config_path='../../config/config.toml')
        hitl_agent_instance = AgentManager().get_instance_obj('HITL_agent')
        agent_universe_initialized = True


def chat(question: str):
    hitl_agent_instance : Agent = AgentManager().get_instance_obj('HITL_agent')
    output_object: OutputObject = hitl_agent_instance.run(input=question, session_id='test_session')
    return output_object


#   清除历史记录按钮
def click_button():
    st.session_state.clicked = True


# 向页面添加agent消息
def display_message(role, content):
    st.session_state["user_content"].append({"role": role, "content": f"{role}:{content}"})
    with st.chat_message(role, avatar=avatar_icon[role]):
        st.markdown(f"{role}:{content}")


def main():
    st.set_page_config(page_title="HITL", page_icon="🤖")
    st.title(" HITL Agent")
    st.sidebar.button(label='🧹清除历史记录', on_click=click_button)  # 设置清除按钮

    if 'HITL_agent_initialized' not in st.session_state:
        initialize_agent_universe_and_agent()
        st.session_state['HITL_agent_initialized'] = True

    if "user_content" not in st.session_state:
        st.session_state["user_content"] = []

    if "clicked" in st.session_state and st.session_state["clicked"]:  # 清除历史记录逻辑
        st.session_state["user_content"] = []
        st.session_state["clicked"] = False

    for message in st.session_state["user_content"]:
        with st.chat_message(message["role"], avatar=avatar_icon[message["role"]]):
            st.markdown(message["content"])

        # Get user's question
    if user_question := st.chat_input("请输入你的问题...", max_chars=2120):
        # 显示用户问题
        st.session_state["user_content"].append({"role": "user", "content": user_question})
        st.chat_message("user_content", avatar=avatar_icon["user"]).markdown(user_question)

        # 程序入口，调用chat函数与agent交互
        output_object = chat(user_question)

        # 获取每个agent的回答

        Client_Manager_Result = output_object.get_data("output").get("client_manager_result")
        slot_full = Client_Manager_Result.get_data("slot_full")
        if slot_full:
            # 由于client manager agent 需要和人交互，因此根据slot_full单独处理
            client_manager_result = Client_Manager_Result.get_data("framework")
            cm_res = "\n"
            for index, one_framework in enumerate(client_manager_result):
                cm_res += f"[{index + 1}] {one_framework} \n"
            client_manager_result = cm_res

            # Task_Excute_One_Result = output_object.get_data("output").get("task_excute_one_agent_result")
            # task_excute_one_result: OutputObject = Task_Excute_One_Result.get_data("task_excute_one_agent_result").get(
            #     "output")
            
            # Task_Excute_Two_Result = output_object.get_data("output").get("task_excute_two_agent_result")
            # task_excute_two_result: OutputObject = Task_Excute_Two_Result.get_data("task_excute_two_agent_result").get(
            #     "output")
            
            # Task_Excute_Three_Result = output_object.get_data("output").get("task_excute_three_agent_result")
            # task_excute_three_result: OutputObject = Task_Excute_Three_Result.get_data("task_excute_three_agent_result").get(
            #     "output")
            Task_Excute_One_Result = output_object.get_data("output").get("task_result")
            task_excute_one_result = Task_Excute_One_Result.get('task_excute_one_agent_result').get('output')

            Task_Excute_Two_Result = output_object.get_data("output").get("task_result")
            task_excute_two_result = Task_Excute_Two_Result.get('task_excute_two_agent_result').get('output')

            Task_Excute_Three_Result = output_object.get_data("output").get("task_result")
            task_excute_three_result = Task_Excute_Three_Result.get('task_excute_three_agent_result').get('output')
            
            System_Architect_Result = output_object.get_data("output").get("system_architect_result")
            system_architect_result: OutputObject = System_Architect_Result.get_data("system_architect_result").get(
                "output")

            Dispatcher_Agent_Result = output_object.get_data("output").get("dispatcher_agent_result")
            dispatcher_agent_result: OutputObject = Dispatcher_Agent_Result.get_data("dispatcher_agent_result").get(
                "output")

            Operation_And_Maintenance_Engineer_Result = output_object.get_data("output").get(
                "operation_and_maintenance_engineer_result")
            # 输出格式: {'output': {'suggestion': 'xx','is_useful',True/False},'score': 80 ,'suggestion':'xx'}
            operation_and_maintenance_engineer_result: OutputObject = Operation_And_Maintenance_Engineer_Result.get_data(
                "operation_and_maintenance_engineer_result").get("output").get("suggestion")

            display_message("client_manager", client_manager_result)
            display_message("task_execute_one_agent", task_excute_one_result)
            display_message("task_execute_two_agent", task_excute_two_result)
            display_message("task_execute_three_agent", task_excute_three_result)
            display_message("system_architect", system_architect_result)
            display_message("dispatcher_agent", dispatcher_agent_result)
            display_message("operation_and_maintenance_engineer_agent", operation_and_maintenance_engineer_result)
        else:
            client_manager_result = Client_Manager_Result.get_data("output")
            display_message("client_manager", client_manager_result)


if __name__ == "__main__":
    main()
