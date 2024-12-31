"""HITL planner module."""
from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.agent_manager import AgentManager
from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject
from agentuniverse.agent.plan.planner.planner import Planner
from agentuniverse.base.util.logging.logging_util import LOGGER



default_sub_agents = {
    'CMA': 'ClientManagerPlanner',
    'TE_ONE': 'TaskExcuteOneAgent',
    'TE_TWO': 'TaskExcuteTwoAgent',
    'TE_THREE': 'TaskExcuteThreeAgent',
    'SAA': 'SystemArchitectAgent',
    'OAM': 'OperationAndMaintenanceEngineerAgent',
    'DA': 'DispatcherAgent',
}


default_jump_step = ''  # 可以选择跳过的智能体

default_eval_threshold = 60

default_retry_count = 3




class HITLPlanner(Planner):
    """Peer planner class."""

    def invoke(self, agent_model: AgentModel, planner_input: dict, input_object: InputObject) -> dict:
        """Invoke the planner.

        Args:
            agent_model (AgentModel): Agent model object.
            planner_input (dict): Planner input object.
            input_object (InputObject): The input parameters passed by the user.
        Returns:
            dict: The planner result.
        """
        planner_config = agent_model.plan.get('planner')
        sub_agents = self.generate_sub_agents(planner_config)
        return self.agents_run(sub_agents, planner_config, planner_input, input_object)

    @staticmethod
    def generate_sub_agents(planner_config: dict) -> dict:
        """Generate sub agents.

        Args:
            planner_config (dict): Planner config object.
        Returns:
            dict: Planner agents.
        """
        agents = dict()
        for config_key, default_agent in default_sub_agents.items():
            config_data = planner_config.get(config_key, None)
            if config_data == '':
                continue
            agents[config_key] = AgentManager().get_instance_obj(config_data if config_data else default_agent)
        return agents

    @staticmethod
    def build_expert_framework(planner_config: dict, input_object: InputObject):
        """Build expert framework for the given planner config object.

        Args:
            planner_config (dict): Planner config object.
            input_object (InputObject): Agent input object.
        """
        expert_framework = planner_config.get('expert_framework')
        if expert_framework:
            context = expert_framework.get('context')
            selector = expert_framework.get('selector')
            if selector:
                selector_result = ToolManager().get_instance_obj(selector).run(**input_object.to_dict())
                input_object.add_data('expert_framework', selector_result)
            elif context:
                input_object.add_data('expert_framework', context)

    def agents_run(self, agents: dict, planner_config: dict, agent_input: dict, input_object: InputObject) -> dict:
        """Planner agents run.

        Args:
            agents (dict): Planner agents.
            planner_config (dict): Planner config object.
            agent_input (dict): Planner input object.
            input_object (InputObject): Agent input object.
        Returns:
            dict: The planner result.
        """
        result: dict = dict()

        loopResults = list()

        client_manager_result = dict()
        system_architect_result = dict()
        operation_and_maintenance_engineer_result = dict()
        dispatcher_agent_result = dict()
        
        task_excute_one_agent_result = dict()
        task_excute_two_agent_result = dict()
        task_excute_three_agent_result = dict()

        retry_count = planner_config.get('retry_count', default_retry_count)  # -> 在HITL_agent.yaml中配置,未配置则使用默认值
        jump_step = planner_config.get('jump_step', default_jump_step)
        eval_threshold = planner_config.get('eval_threshold', default_eval_threshold)

        self.build_expert_framework(planner_config, input_object)

        clientManagerAgent: Agent = agents.get('CMA')
        systemArchitectAgent: Agent = agents.get('SAA')
        operationAndMaintenanceEngineerAgent: Agent = agents.get('OAM')
        dispatcherAgent: Agent = agents.get('DA')

        taskExcuteOneAgent: Agent = agents.get('TE_ONE')
        taskExcuteTwoAgent: Agent = agents.get('TE_TWO')
        taskExcuteThreeAgent: Agent = agents.get('TE_THREE')

    

   

        LOGGER.info(f"Starting Account Manager agent.")
        if not client_manager_result or jump_step == "CMA":
            if not clientManagerAgent:
                LOGGER.warn("no Account Manager agent.")
                client_manager_result = OutputObject({"framework": [agent_input.get('input')]})
            else:
                LOGGER.info(f"Starting Account Manager agent.")
                client_manager_result = clientManagerAgent.run(**input_object.to_dict())
                slot_full = client_manager_result.get_data("slot_full")
                if not slot_full:
                    # 词槽未填充完整，跳过后续agent
                    slot_result = {"client_manager_result": client_manager_result}
                    return {'result': [slot_result]}
            input_object.add_data('client_manager_result', client_manager_result)
            # add client Manager Agent log info
            logger_info = f"\nAccount Manager agent execution result is :\n"
            #logger_info += client_manager_result.get_data('framework')
            # framework 是子问题列表
            for index, one_framework in enumerate(client_manager_result.get_data('framework')):
                logger_info += f"[{index + 1}] {one_framework} \n"
            LOGGER.info(logger_info)
            
            agent_mapping = {
                '政治': agents.get('TE_ONE'),
                '经济': agents.get('TE_TWO'),
                '文化': agents.get('TE_THREE'),
                # 添加更多的键和对应的 agent 映射
            }
            
            # 从任务执行智能体--系统架构师开始，要循环，直到达到评估阈值，或者达到最大重试次数
            for i in range(retry_count):


                task_result = dict()
                for item in client_manager_result.get_data('framework'):
                    for k, v in item.items():
                        selected_agent = agent_mapping.get(k, agents.get('TE_ONE'))  # 使用 TE_ONE 作为默认 agent
                        if selected_agent:
                            #LOGGER.info(f"Starting {selected_agent}.")
                            res = selected_agent.run(**input_object.to_dict())
                            task_result.update(res.to_dict())
                        else:
                            LOGGER.info(f"Error When Selected Agent")
                input_object.add_data('task_result', task_result)

                # LOGGER.info(f"Starting iteration {i + 1} of {retry_count}")
                
                # if not taskExcuteOneAgent:
                #     LOGGER.warn('no Task Execute One agent.')
                #     task_excute_one_agent_result = OutputObject({})
                # else:
                #     LOGGER.info(f"Starting Task Execute One agent.")
                #     task_excute_one_agent_result = taskExcuteOneAgent.run(**input_object.to_dict())
                
                # input_object.add_data('task_excute_one_agent_result', task_excute_one_agent_result)
                # # add expressing agent log info
                # logger_info = f"\nTask Execute One agent result is :\n"
                # logger_info += f"{task_excute_one_agent_result.get_data('task_excute_one_agent_result').get('output')}"
                # LOGGER.info(logger_info)
                
                # if not taskExcuteTwoAgent:
                #     LOGGER.warn('no Task Execute Two agent.')
                #     task_excute_two_agent_result = OutputObject({})
                # else:
                #     LOGGER.info(f"Starting Task Execute Two agent.")
                #     task_excute_two_agent_result = taskExcuteTwoAgent.run(**input_object.to_dict())
                
                # input_object.add_data('task_excute_two_agent_result', task_excute_two_agent_result)
                # # add expressing agent log info
                # logger_info = f"\nTask Execute Two agent result is :\n"
                # logger_info += f"{task_excute_two_agent_result.get_data('task_excute_two_agent_result').get('output')}"
                # LOGGER.info(logger_info)

                # if not taskExcuteThreeAgent:
                #     LOGGER.warn('no Task Execute Three agent.')
                #     task_excute_three_agent_result = OutputObject({})
                # else:
                #     LOGGER.info(f"Starting Task Execute Three agent.")
                #     task_excute_three_agent_result = taskExcuteThreeAgent.run(**input_object.to_dict())
                
                # input_object.add_data('task_excute_three_agent_result', task_excute_three_agent_result)
                # # add expressing agent log info
                # logger_info = f"\nTask Execute Three agent result is :\n"
                # logger_info += f"{task_excute_three_agent_result.get_data('task_excute_three_agent_result').get('output')}"
                # LOGGER.info(logger_info)


                # if not system_architect_result or jump_step in ["CMA", "SAA"]:
                if not systemArchitectAgent:
                    LOGGER.warn("no System Architect agent.")
                    system_architect_result = OutputObject({})
                else:
                    LOGGER.info(f"Starting System Architect agent.")
                    system_architect_result = systemArchitectAgent.run(**input_object.to_dict())

                input_object.add_data('system_architect_result', system_architect_result)
                # add expressing agent log info
                logger_info = f"\nSystem Architect agent execution result is :\n"
                logger_info += f"{system_architect_result.get_data('system_architect_result').get('output')}"
                LOGGER.info(logger_info)

                #if not dispatcher_agent_result or jump_step in ["CMA", "SAA", "DA"]:
                if not dispatcherAgent:
                    LOGGER.warn("no Scheduler agent.")
                    dispatcher_agent_result = OutputObject({})
                else:
                    LOGGER.info(f"Starting Scheduler agent.")
                    dispatcher_agent_result = dispatcherAgent.run(**input_object.to_dict())

                input_object.add_data('dispatcher_agent_result', dispatcher_agent_result)
                logger_info = f"\nScheduler agent execution result is :\n"
                logger_info += f"{dispatcher_agent_result.get_data('dispatcher_agent_result').get('output')}"
                LOGGER.info(logger_info)

                #if not operation_and_maintenance_engineer_result or jump_step in ["CMA", "SAA", "DA", "OAM"]:
                if not operationAndMaintenanceEngineerAgent:
                    LOGGER.warn("no Operation And Maintenance Engineer agent.")
                    # operation_and_maintenance_engineer_result = OutputObject({})
                    loopResults.append({
                        "client_manager_result": client_manager_result,
                        "task_result":task_result,
                        "system_architect_result": system_architect_result,
                        "dispatcher_agent_result": dispatcher_agent_result,
                        "operation_and_maintenance_engineer_result": operation_and_maintenance_engineer_result
                    })
                    result["result"] = loopResults
                    return result
                else:
                    LOGGER.info(f"Starting Operation And Maintenance Engineer agent.")
                    operation_and_maintenance_engineer_result = operationAndMaintenanceEngineerAgent.run(
                        **input_object.to_dict())

                    input_object.add_data('operation_and_maintenance_engineer_result',
                                            operation_and_maintenance_engineer_result)
                    logger_info = f"\nOperation And Maintenance Engineer agent execution result is :\n"
                    logger_info += f"Suggestion: {operation_and_maintenance_engineer_result.get_data('operation_and_maintenance_engineer_result').get('suggestion')} \n"
                    logger_info += f"Score: {operation_and_maintenance_engineer_result.get_data('operation_and_maintenance_engineer_result').get('score')}\n"
                    LOGGER.info(logger_info)

                    if operation_and_maintenance_engineer_result.get_data('operation_and_maintenance_engineer_result').get('score') and operation_and_maintenance_engineer_result.get_data('operation_and_maintenance_engineer_result').get('score') >= eval_threshold:
                        # OAM的评价分大于阈值 则返回结果
                        loopResults.append({
                            "client_manager_result": client_manager_result,
                            "task_result":task_result,
                            "system_architect_result": system_architect_result,
                            "dispatcher_agent_result": dispatcher_agent_result,
                            "operation_and_maintenance_engineer_result": operation_and_maintenance_engineer_result
                        })
                        result["result"] = loopResults
                        return result
                    else:
                        loopResults.append({
                            "client_manager_result": client_manager_result,
                            "task_result":task_result,
                            #"task_excute_one_agent_result": task_excute_one_agent_result,
                            #"task_excute_two_agent_result": task_excute_two_agent_result,
                            #"task_excute_three_agent_result": task_excute_three_agent_result,
                            "system_architect_result": system_architect_result,
                            "dispatcher_agent_result": dispatcher_agent_result,
                            "operation_and_maintenance_engineer_result": operation_and_maintenance_engineer_result
                        })

                        continue

        result["result"] = loopResults
        return result
