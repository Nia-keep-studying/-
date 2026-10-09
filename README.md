# 顾客售后客服助手

服务一个虚构商家的学习与作品项目，使用模拟订单和示例制度。当前是综合作品的起点：复用已经跑通过的订单 API、混合检索、重排序与 LangChain 会话循环，后续在这个仓库逐步实现顾客身份、订单归属和人工工单。

## 当前能力与下一步

- 已迁移：订单 CRUD API；关键词映射与向量召回、合并去重、重排序；查询订单与制度两个工具；同一进程内按 thread_id 隔离会话。
- 尚未实现：登录鉴权、本人订单权限、候选订单确认、业务入口链接、人工工单、会话持久化、上下文压缩、向量数据库。
- 第一版最终目标以 [需求与验收 v0.2](docs/第一版需求说明与验收样例_v0.2.docx) 为准。原无版本后缀的 Word 是历史 v0.1；其中旧方案不作为当前目标。
- 第一版通过业务入口引导顾客操作；提供链接不代表申请已提交、退款已批准或已经到账。人工工单只有保存成功才能告知编号，不表示人工已经接通。

当前 API 保留学习 Demo 的读写端点，尚无鉴权，供本地模拟开发；Agent 只注册查询工具。登录与数据归属将在下一阶段由学习者实现。本项目没有接入真实企业系统。

## 目录与职责

```text
customer-support-assistant/
  agent/
    langchain_agent_demo.py  # 模型、工具包装、Agent 和终端会话循环
    order_client.py          # 访问订单 HTTP API
  backend/
    sqlmodel_server.py       # FastAPI、SQLModel 与订单规则
  rag/
    documents/              # 3 份公开示例制度
    rag_loader.py           # 读取文档、按章节切片
    rag_search.py           # 关键词映射、检索、合并去重
    embedding_demo.py       # 嵌入模型、片段向量与相似度检索
    rerank_demo.py           # 重排序模型与排序函数
    policy_search.py         # 两路召回 → 去重 → 重排序 → 返回原文
    eval_retrieval.py        # 原有三题召回阶段评估
  docs/                     # 需求说明与验收样例
  tests/                    # 离线迁移检查
  settings.py               # 根目录、环境变量与检索参数
  requirements.txt
  .env.example
```

backend、agent 和 rag 是同一个仓库中的 Python 包；订单 API 与 Agent 仍运行在两个进程。所有命令从仓库根目录执行，使用 `python -m`，无需 sys.path 修改或旁边的旧 Demo。原 order-agent 与 rag-demo 保留作学习阶段记录，后续综合项目功能在本仓库维护。

Agent 的 `main()` 只加载一次检索资源，再通过 `build_agent(...)` 参数传给工具包装。工具函数使用这些已经准备好的对象，框架没有自动帮我们加载文档或注入模型。

## 环境与启动（PowerShell）

使用 Python 3.12。新克隆环境按以下步骤准备：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

第一次复制后，手动在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`。已有 `.env` 时不要再次复制覆盖。首次运行会下载两个模型到 `.model_cache`；模型推理在本机 CPU 执行，DeepSeek 回答通过 API 调用并产生相应费用。原项目的密钥、数据库、模型缓存和虚拟环境没有搬进新目录。

终端 1 启动订单 API，使用 8001 端口与旧 Demo 的 8000 区分：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.sqlmodel_server:app --host 127.0.0.1 --port 8001
```

启动后生成新的 `data/orders_sqlmodel.db`，不会读取旧 Demo 的订单数据库。浏览器打开 http://127.0.0.1:8001/docs ，通过 `POST /orders` 创建一笔演示订单：

```json
{"order_id": "A1001", "product": "演示商品", "status": "已发货"}
```

新数据库初始为空；重复创建同一编号返回 409。终端 2 启动 Agent：

```powershell
.\.venv\Scripts\python.exe -m agent.langchain_agent_demo
```

可输入 `A1001 已经发货，还能修改收货地址吗？`，观察订单与制度工具调用及来源。输入 `新会话` 更换 thread_id，输入 `q` 退出。InMemorySaver 仅保留当前进程内的历史，退出后不会恢复；消息数包含工具调用和工具返回，不等于问题数或 token 数。

当前本机也可以临时使用原 rag-demo 的 Python 环境运行上述模块，不需要重新安装。新仓库没有依赖旧目录的 requirements 引用。若希望复用已下载模型，可自行将 `.env` 的 MODEL_CACHE_DIR 指向旧缓存的绝对路径；该机器路径不提交到仓库。

## 检查与评估

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m rag.rag_loader
.\.venv\Scripts\python.exe -m rag.eval_retrieval
```

离线测试使用临时数据库及模拟模型，验证包导入、API 边界、候选合并、重排序输出和工具消息配对，不调用付费 API。原三题评估会加载本地嵌入模型，输出到 `data/evaluation_results.json`；它只评估重排序前的召回阶段，不代表完整回答质量。无依据题有候选片段也可能最终正确拒答，应分别观察召回噪声与回答是否有依据。

检索当前使用 VECTOR_TOP_K=5、MIN_SCORE=0.4、RERANK_TOP_K=3，集中在 settings.py。这是此前实验设置，不是通用最优值。示例文档不包含电子发票制度，模型应说明缺少依据。

2026-10-09 迁移验证：5 项离线检查通过；pip check 无依赖冲突，独立 requirements.txt 在当前环境的离线安装预检通过。复用原缓存、关闭联网后，实际嵌入与重排序模型加载成功，12 个片段得到 (12, 64) 向量；“发货后能修改收货地址吗？”的选中结果包含地址修改。没有调用付费回答 API；干净环境安装及新入口的真实模型回答仍待验收。
