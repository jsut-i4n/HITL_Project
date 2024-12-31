def select_agent(task, agents):
    # 这里定义了每个 agent 处理的任务条件
    conditions = {
        'TE_ONE': lambda x: x.get('k1') == 'value_for_one',
        'TE_TWO': lambda x: x.get('k2') == 'value_for_two',
        'TE_THREE': lambda x: x.get('k3') == 'value_for_three'
    }

    # 遍历 conditions，找到第一个满足条件的 agent
    for agent_key, condition in conditions.items():
        if condition(task):
            return agents.get(agent_key)

    return agents.get('TE_ONE')