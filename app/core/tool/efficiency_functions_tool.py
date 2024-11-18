from typing import Optional

from pydantic import Field
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from agentuniverse.agent.action.tool.tool import Tool, ToolInput
from agentuniverse.base.config.component_configer.configers.tool_configer import ToolConfiger
from agentuniverse.base.util.env_util import get_from_env
import re


class EfficiencyFunctionTool(Tool):
    """
    Efficiency Function Tool 

    """

    def execute(self, tool_input: ToolInput):
        input = tool_input.get_data("dispatcher_agent_result")
        try:
            inputs = input.get_data("dispatcher_agent_result").get("output")
            pattern_w_t = r'[Ww]_t\s*[=：:]\s*(\d+\.\d+)\s*(?:\([^\)]*\))?'
            pattern_w_e = r'[Ww]_e\s*[=：:]\s*(\d+\.\d+)\s*(?:\([^\)]*\))?'

            match_w_e = re.search(pattern_w_e, inputs)
            match_w_t = re.search(pattern_w_t, inputs)

            if match_w_t:
                # 提取匹配的数字部分并转换为浮点数
                w_t = float(match_w_t.group(1))
            else:
                w_t = None
                print("w_t was not found in the text.")

            if match_w_e:
                # 提取匹配的数字部分并转换为浮点数
                w_e = float(match_w_e.group(1))
            else:
                w_e = None
                
                print("w_e was not found in the text.")

            results = None
            if w_e is not None and w_t is not None:
           
                results = w_e * w_t
                print("efficiency functions tool result:", results)
            else:
                print("Cannot calculate results due to missing values.")

            # 工具调用结果应该是str类型
            if isinstance(results,(int,float)):
                results = str(results)
                results += "效能函数工具调用结果为：w_t*w_e="+ results + "\n"
            return results
        except Exception as e:
            print(f"Error: Failed to use efficiency functions tool.   + {str(e)}")
            return ''

