# 顾客售后客服助手

服务一个虚构商家的学习与作品项目，使用模拟订单和示例制度。复用订单 API、混合检索、重排序与 LangChain 会话循环；当前已实现后端登录会话和本人订单读取，接下来整理旧写接口并把凭证接入 Agent，再实现业务引导与人工工单。

## 当前能力与下一步

- 已迁移：订单 CRUD API；关键词映射与向量召回、合并去重、重排序；查询订单与制度两个工具；同一进程内按 thread_id 隔离会话。
- 已实现：测试账号登录、随机凭证签发、凭证哈希存储、到期检查、本人订单列表与详情、退出登录撤销当前会话。
- 尚未实现：Agent 请求携带顾客凭证、候选订单确认、业务入口链接、人工工单、会话持久化、上下文压缩、向量数据库、定期清理过期会话。
- 第一版最终目标以 [需求与验收 v0.3](docs/第一版需求说明与验收样例_v0.3.docx) 为准。v0.1 与 v0.2 保留作历史记录。
- 第一版通过业务入口引导顾客操作；提供链接不代表申请已提交、退款已批准或已经到账。人工工单只有保存成功才能告知编号，不表示人工已经接通。

`GET /me`、`GET /orders`、`GET /orders/{order_id}` 和 `POST /logout` 已要求有效凭证；列表与详情只读取当前用户的订单。旧的订单 POST/PATCH/PUT/DELETE 路由仍保留，未加操作权限检查，创建订单也缺少必填 user_id 的适配；它们不是本轮验收范围，下一步撤下顾客服务中的通用写路由。当前仅供本地模拟开发，不能作为完整权限保护的服务部署。

订单创建、支付、发货和退款执行由订单或售后系统负责，不属于第一版助手。当前通过 seed_demo 提供模拟上游订单；Agent 只注册查询工具，但 HTTP 客户端尚未发送凭证，因此现在直接运行 Agent 查询受保护订单会收到 401，不能把此前迁移版本的整合验收当作本轮鉴权后的端到端验收。本项目没有接入真实企业系统。

## 目录与职责

```text
customer-support-assistant/
  agent/
    langchain_agent_demo.py  # 模型、工具包装、Agent 和终端会话循环
    order_client.py          # 访问订单 HTTP API
  backend/
    sqlmodel_server.py       # FastAPI、SQLModel 与订单规则
    seed_demo.py             # 初始化两位模拟顾客及三笔订单
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

先初始化演示数据库：

```powershell
.\.venv\Scripts\python.exe -m backend.seed_demo
```

脚本创建 `data/orders_sqlmodel.db`，其中 U1001（alice）拥有 A1001、A1002，U1002（bob）拥有 A2001，三笔订单均已发货。密码通过 pwdlib/Argon2 生成哈希后保存；演示密码分别为 `demo-password` 和 `demo_password_2`。重复运行会跳过已存在的编号，不覆盖已有记录。初始化脚本通过模块方式运行，不直接执行 backend/seed_demo.py。

终端 1 启动订单 API，使用 8001 端口与旧 Demo 的 8000 区分：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.sqlmodel_server:app --host 127.0.0.1 --port 8001
```

浏览器打开 http://127.0.0.1:8001/docs ，按以下顺序验证后端：

1. 调用 `POST /login`，请求正文为 `{"username":"alice","password":"demo-password"}`。
2. 从响应复制 `access_token`，点击 Authorize，只粘贴原始凭证。客户端请求头为 `Authorization: Bearer <token>`。
3. 调用 `/me` 得到当前用户；Alice 的列表只有 A1001/A1002，访问 Bob 的 A2001 返回 404。替换为 Bob 的凭证后列表只有 A2001。
4. 调用 `/logout` 返回 204，保留旧凭证再次查询应返回 401；重新登录取得新凭证后恢复。

凭证保存在数据库中的形式是 SHA-256 哈希，用户密码使用 pwdlib/Argon2；身份验证不会把密码哈希返回客户端。有效期由 `LOGIN_SESSION_TTL_SECONDS` 控制，当前为一小时。过期记录即使尚未删除，也不能通过验证；修改有效期只影响之后签发的会话。到期测试已用一分钟配置完成，随后恢复一小时。

数据库独立于旧 Demo；已有数据库不会因为 Python 模型修改而自动迁移。当前使用带 user_id 的新项目数据库。原通用创建接口不要用于准备数据。

框架入口保留下面的运行方式，鉴权后的真实订单调用须等待客户端凭证接入完成：

```powershell
.\.venv\Scripts\python.exe -m agent.langchain_agent_demo
```

后续接入凭证后，可用 `A1001 已经发货，还能修改收货地址吗？` 验证订单与制度工具及来源；当前该订单工具因缺少凭证会失败。输入 `新会话` 更换 thread_id，输入 `q` 退出。InMemorySaver 仅保留当前进程内的历史，退出后不会恢复；消息数包含工具调用和工具返回，不等于问题数或 token 数。thread_id 不代替登录身份，后续还需把会话归属绑定到可信用户。

当前本机也可以临时使用原 rag-demo 的 Python 环境运行上述模块，不需要重新安装。新仓库没有依赖旧目录的 requirements 引用。若希望复用已下载模型，可自行将 `.env` 的 MODEL_CACHE_DIR 指向旧缓存的绝对路径；该机器路径不提交到仓库。

## 检查与评估

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m rag.rag_loader
.\.venv\Scripts\python.exe -m rag.eval_retrieval
```

离线测试使用临时数据库及模拟模型，不修改演示数据库，不调用付费 API。检查覆盖账号密码、凭证哈希存储、未登录拒绝、用户订单隔离、当前会话撤销、其他会话保留、到期边界、初始化重复执行，以及包导入、检索和工具消息配对。旧的未登录 CRUD 测试已随数据模型改为初始化检查；通用写路由的兼容和权限尚未验收，测试通过不表示这些路由已完成保护。

原三题评估会加载本地嵌入模型，输出到 `data/evaluation_results.json`；它只评估重排序前的召回阶段，不代表完整回答质量。无依据题有候选片段也可能最终正确拒答，应分别观察召回噪声与回答是否有依据。

检索当前使用 VECTOR_TOP_K=5、MIN_SCORE=0.4、RERANK_TOP_K=3，集中在 settings.py。这是此前实验设置，不是通用最优值。示例文档不包含电子发票制度，模型应说明缺少依据。

2026-10-09 迁移验证：5 项离线检查通过；pip check 无依赖冲突，独立 requirements.txt 在当前环境的离线安装预检通过。复用原缓存、关闭联网后，实际嵌入与重排序模型加载成功，12 个片段得到 (12, 64) 向量；“发货后能修改收货地址吗？”的选中结果包含地址修改。没有调用付费回答 API；干净环境安装及新入口的真实模型回答仍待验收。

随后增加用户与归属字段：seed_demo 实际运行两次后仍为 2 位用户、3 笔订单，归属和演示密码验证通过。上述 5 项迁移测试记录属于增加字段前的版本，原 API 测试的创建数据需要适配；不代表当前鉴权或全部端点已经验收。

2026-10-10 鉴权收尾：学习者实际验证本人订单读取、退出后旧凭证 401、重新登录恢复，以及 60 秒自然到期后订单查询 401。助手补充并更新离线检查，当前 9 项测试通过，pip check 无冲突；过期边界采用控制时间验证，没有等待或调用模型。独立干净环境安装和鉴权后 Agent 的真实回答仍待验收。
