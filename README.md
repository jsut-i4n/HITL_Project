### **项目前的必要准备**<br>

- 安装pycharm或vscode。
- 安装redis数据库。

- 安装[Ollama](https://ollama.com/),根据自己的需要，下载要使用的llm模型文件。
- 创建虚拟环境 `conda create -n au python=3.10`
  - 在虚拟环境下使用`pip install -r requirements.txt`安装项目所需的包。
---
**目录简介**<br>

```
├── HITL_Project  
│   ├── app  
│   │   ├── core  
│   │   │   ├── agent  
│   │   │   ├── knowledge  
│   │   │   ├── llm  
│   │   │   ├── memory  
│   │   │   ├── planner  
│   │   │   ├── prompt 
│   │   │   ├── scene_config 
│   │   │   └── tool    
│   │   ├── example 
│   │   │   ├── .test_log_dir
│   │   │   ├── assets
│   │   │   ├── monitor  
│   │   │   └── my_agent.py  
│   ├── config  
│   │   ├── config.toml  
│   │   ├── custom_key.toml  
│   │   └── log_config.toml  
│   ├── DB
│   ├── resources   
│   └── README.md
```

***app***文件夹包括了项目的核心文件夹core以及使用样例example。<br>

core核心文件夹里面包括了agent、knowledge、llm、memory、planner、prompt、tool。<br>

`agent`：搭建的智能体<br>

`knowledge`：智能体拥有的知识<br>

`llm`：agent使用的llm。如果需要修改使用的llm模型，在该处修改。<br>

`memory`：智能体的记忆<br>

`planner`：智能体的工作流程<br>

`scene_config`：意图识别、词槽等场景配置相关模块<br>

`prompt`：agent的提示词，包括agent的身份的描述introduction、agent的目标target、以及agent的指示instruction。<br>

`tool`：智能体使用的工具<br>

---

***example***文件夹下的monitor记录了agent每次执行的中间过程<br>

---

***config***文件里面包括了<br>

- 1.`config.toml` ：基本的包路径、配置文件路径以及监控相关的配置。

- 2.`custom_key.toml`：用户的api_key配置，如调用调用DASHSCOPE、设置redis的地址REDIS_URL等。
  ```toml
  # Example file of custom_key.toml. Rename to custom_key.toml while using.
  [KEY_LIST]
  # Perform a full component scan and registration for all the paths under this list.
  example_key = 'AnExampleKey'
  SERPER_API_KEY=''  # 谷歌搜索API  可以去serper官网注册
  OPENAI_API_KEY='YourOpenAIKey'
  DASHSCOPE_API_KEY=''   # 调用DASHSCOPE API 可以去阿里的灵积官网注册
  
  OLLAMA_BASE_URL="http://localhost:11434"   # 默认值
  
  REDIS_URL="redis://localhost:6379/0"  #redis数据库的url  地址:端口/数据库
  ```

- 3.`log_config.toml`：保存的日志的设置。

---

***DB***文件夹用来存储knowledge中加载的PDF等embedding后的知识文档<br>

---

### **开始**<br>

1.首先在终端启动redis，用于存储大模型输出的数据。<br>

- cd 到你自己的redis安装目录
- 使用如下命令启动redis的服务模式
```shell
redis-server &
```
- redis-server命令后面加上&符号，表示后台运行redis服务。

- <font color=red>注意：如果你是在linux系统上运行，需要在前面加上**sudo**命令，使redis有权限保存llm交互的数据。</font>


2.再打开一个终端，使用如下命令启动对应的模型，qwen2:7b指运行的模型名称<br>
```shell
ollama run qwen2:7b
```
- 命令 `ollama run qwen:7b` 就是在ollama环境下启动qwen2:7b模型
- 如果你换成了自己的模型，还需要在**app/core/llm**文件下修改**ollama_llm.yaml**文件,需要将**model_name**替换成你自己使用的模型名称。
  ```yaml
    name: 'ollama_llm'
    description: '调用本地的ollama环境下的LLM模型'
    model_name: 'qwen2:7b'  # qwen2:72b
    max_tokens: 1000
    max_context_length: 32000
    metadata:
      type: 'LLM'
      module: 'HITL_Project.app.core.llm.ollama_llm'
      class: 'OllamaLLM'  # 类名
  ```

3.以**streamlit**方式运行目录**HITL_Project\app\example**文件下的my_agent.py文件启动agent进行问答。<br>

- 在pycharm或vscode的终端输入如下3条命令，以通过streamlit启动my_agent.py脚本文件。
- 如果已经激活了au虚拟环境，当前目录也在**HITL_Project\app\example**目录下，可以省去前两条命令。
- 激活虚拟环境 `conda activate au`
- ```shell
  cd HITL_Project\app\example
  ```
- ```shell
  streamlit run my_agent.py
  ```



