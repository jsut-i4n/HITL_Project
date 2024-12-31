import redis
import json


# 连接到Redis
def get_chat_history(session_id, msg_num):
    """
    获取redis中聊天历史
    :param msg_num:
    :param session_id:
    :return:
    """
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)

    # 从Redis中获取数据
    data = r.lrange('message_store:'+session_id, 0, msg_num)

    # 解析JSON字符串
    decoded_data = [json.loads(item.decode('utf-8')) for item in data]

    result_list = [
        {"type": item["type"], "content": item["data"]["content"]}
        for item in decoded_data
    ]
    return result_list




