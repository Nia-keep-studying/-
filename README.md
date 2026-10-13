# 顾客售后客服助手

服务一个虚构商家的学习与作品项目，使用模拟订单和示例制度。复用订单 API、混合检索、重排序与 LangChain 会话循环；当前已接通登录身份、本人订单查询、候选订单确认、制度回答与退货演示入口。人工工单后端已写，Agent 接入仍待完成。

## 当前能力与下一步

- 已复用：关键词映射与向量召回、合并去重、重排序；同一进程内按 thread_id 隔离会话。
- 已实现：测试账号登录、随机凭证签发、凭证哈希存储、到期检查、本人订单列表与详情、退出登录撤销当前会话；终端 Agent 携带凭证、invoke 前身份检查、状态和商品关键词筛选、多笔候选确认。
- 已注册：query_order、query_orders、search_policy、query_return_entry 四个工具。原始凭证由程序传给工具实现，不是模型可填写的工具参数。
- 入口准备：business_entry.get_return_entry 已检查本人订单和已发货场景，按配置生成带 order_id 的演示 URL，保留错误结果；不把已发货视为退款审批结论。
- 已接通：退货演示页可登录并读取本人订单；入口工具已注册，两条对话分支有学习者实测。POST /human-tickets 保存待处理工单，使用可信用户身份并检查关联订单归属，允许不关联订单。
- 尚未实现：工单 HTTP 客户端与 Agent 工具、统一网页登录及聊天、会话持久化与归属检查、上下文压缩、向量数据库、定期清理过期会话、完整翻页策略和 has_more。
- 第一版最终目标以 [需求与验收 v0.6](docs/第一版需求说明与验收样例_v0.6.docx) 为准。旧版本保留作历史记录。
- 第一版通过业务入口引导顾客操作；提供链接不代表申请已提交、退款已批准或已经到账。人工工单只有保存成功才能告知编号，不表示人工已经接通。

业务接口为 `POST /login`、`GET /me`、`POST /logout`、`GET /orders`、`GET /orders/{order_id}`、`POST /human-tickets`。登录之外均要求有效凭证；订单读取和工单关联都限制当前用户归属。`GET /demo/return-application` 是公开 HTML 外壳，隐藏于 OpenAPI，订单数据仍需登录。旧通用订单 POST/PATCH/PUT/DELETE 已移除。当前仅供本地模拟开发，完整网页鉴权、会话归属、部署与运行治理仍待补。

订单创建、支付、发货和退款执行由订单或售后系统负责，不属于第一版助手。当前通过 seed_demo 提供模拟上游订单；客户端发送 Authorization 请求头，由后端确认身份与归属。本项目没有接入真实企业系统，也没有实现完整商城。

## 目录与职责

```text
customer-support-assistant/
  agent/
    langchain_agent_demo.py  # 模型、工具包装、Agent 和终端会话循环
    order_client.py          # 访问订单 HTTP API
    business_entry.py        # 订单校验与演示入口URL生成
  backend/
    sqlmodel_server.py       # 登录、本人订单、页面响应与工单保存
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
  web/return_application.html # 退货演示页面，独立网页登录
  tests/                    # 离线迁移检查
  settings.py               # 根目录、环境变量与检索参数
  requirements.txt
  .env.example
```

backend、agent 和 rag 是同一个仓库中的 Python 包；订单 API 与 Agent 仍运行在两个进程。所有命令从仓库根目录执行，使用 `python -m`，无需 sys.path 修改或旁边的旧 Demo。原 order-agent 与 rag-demo 保留作学习阶段记录，后续综合项目功能在本仓库维护。

Agent 的 `main()` 先登录，再只加载一次检索资源，通过 `build_agent(...)` 参数传给工具包装。内层工具使用外层函数提供的资源与凭证，这是闭包中的显式依赖传递，框架没有自动帮我们登录或加载模型。order_client 的 headers 携带身份，params 携带筛选与分页条件，后端从凭证确认 user_id。

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

数据库独立于旧 Demo；已有数据库不会因为 Python 模型修改而自动迁移。当前使用带 user_id 的新项目数据库，示例数据由 seed_demo 准备。

保持订单服务运行，在第二个终端启动 Agent：

```powershell
.\.venv\Scripts\python.exe -m agent.langchain_agent_demo
```

终端输入 alice / demo-password 登录。可依次验证：

1. `查询 A1001 的订单状态`：返回本人订单；`查询 A2001 的订单状态`：未找到可访问订单，不展示 Bob 数据。
2. `帮我查一下手机的订单`：query_orders 使用 product_keyword 找到 A1002，随后可查询详情。
3. 输入 `新会话`，问 `我已经发货的订单还能退款吗？`：列出两笔候选并请用户选择；回复 `手机那一笔` 后结合详情与制度回答，缺条件继续询问。
4. 每轮业务问题进入 invoke 前，程序调用 /me 检查身份；401 提示并退出，连接或其他 HTTP 错误提示后等待下一次输入。验证自然到期时临时把有效期设60秒，重启服务和 Agent，测试后恢复一小时。

输入 `新会话` 更换 thread_id，输入 `q` 退出。InMemorySaver 仅保留当前进程内的历史，退出后不会恢复；消息数包含工具调用和工具返回，不等于问题数或 token 数。thread_id 不代替登录身份，后续还需把会话归属绑定到可信用户。终端版每次启动登录并创建自己的 Agent；网页版应由聊天后端验证请求身份及会话拥有者。

当前详情工具未找到订单时返回通用错误；列表无匹配返回 []，有效期失效返回错误字典。服务故障不应当作空结果。入口登录失败、Agent执行中途的网络错误或凭证到期、完整分页上限仍需完善；invoke 前检查不保证整轮执行期间凭证一直有效，订单 API 仍逐请求验证。

## 业务入口与人工工单

`settings.RETURN_APPLICATION_URL` 由订单服务地址生成 `/demo/return-application`，辅助函数通过 urlencode 加入订单号。Agent 的 query_return_entry 已注册；页面登录后查询订单，凭证只存在页面内存，刷新或新开需重新登录。可打开 `http://127.0.0.1:8001/demo/return-application?order_id=A1002`，用 Alice 登录验证本人订单，再改 A2001 验证拒绝。页面不提供提交售后或执行退款的按钮。

用户订单、适用路径和明确办理意图都已确认时，可直接提供受控入口，不重复询问是否要链接。是否签收需向用户确认，当前订单表没有该字段；未签收按制度解释拒收，签收后才引导退货申请。工具中的状态检查只限定演示场景，不替代申请条件或售后审核。

最终展示计划包含登录页、聊天页、办理演示页，目前只有办理演示页。终端登录的凭证不会自动交给浏览器；完整网页的统一登录和聊天尚未实现。凭证不放URL，会话访问还需验证拥有者。

`POST /human-tickets` 接收 summary、reason、conversation_id，order_id 和 policy_reference 可省略。接口生成编号、可信 user_id、创建时间及待处理状态；有订单号时检查本人归属。无效凭证返回401、不可关联订单404、字段校验失败422，成功保存才返回201及编号。conversation_id 当前仅作关联标识，不支持恢复历史或证明会话归属。工单尚未接入模型工具，没有真实人工接听或客服操作台；重试幂等及状态流转未实现。

当前本机也可以临时使用原 rag-demo 的 Python 环境运行上述模块，不需要重新安装。新仓库没有依赖旧目录的 requirements 引用。若希望复用已下载模型，可自行将 `.env` 的 MODEL_CACHE_DIR 指向旧缓存的绝对路径；该机器路径不提交到仓库。

## 检查与评估

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m rag.rag_loader
.\.venv\Scripts\python.exe -m rag.eval_retrieval
```

离线测试使用临时数据库及模拟模型，不修改演示数据库，不调用付费 API。检查覆盖账号密码、凭证哈希、用户订单隔离、登出和到期、初始化重复执行、业务路由范围、页面响应、工单保存与拒绝不写入，以及包导入、检索、工具消息配对、四项工具注册与可信凭证传递。还检查HTTP过滤、空列表及失效结果，以及invoke前验证失败不会调用Agent。模拟模型检查程序接线，不证明真实模型每次都会正确选择工具。

原三题评估会加载本地嵌入模型，输出到 `data/evaluation_results.json`；它只评估重排序前的召回阶段，不代表完整回答质量。无依据题有候选片段也可能最终正确拒答，应分别观察召回噪声与回答是否有依据。

检索当前使用 VECTOR_TOP_K=5、MIN_SCORE=0.4、RERANK_TOP_K=3，集中在 settings.py。这是此前实验设置，不是通用最优值。示例文档不包含电子发票制度，模型应说明缺少依据。

2026-10-09 迁移验证：5 项离线检查通过；pip check 无依赖冲突，独立 requirements.txt 在当前环境的离线安装预检通过。复用原缓存、关闭联网后，实际嵌入与重排序模型加载成功，12 个片段得到 (12, 64) 向量；“发货后能修改收货地址吗？”的选中结果包含地址修改。没有调用付费回答 API；干净环境安装及新入口的真实模型回答仍待验收。

随后增加用户与归属字段：seed_demo 实际运行两次后仍为 2 位用户、3 笔订单，归属和演示密码验证通过。上述 5 项迁移测试记录属于增加字段前的版本，原 API 测试的创建数据需要适配；不代表当前鉴权或全部端点已经验收。

2026-10-10 鉴权收尾：学习者实际验证本人订单读取、退出后旧凭证 401、重新登录恢复，以及 60 秒自然到期后订单查询 401。助手补充并更新离线检查，当前 9 项测试通过，pip check 无冲突；过期边界采用控制时间验证，没有等待或调用模型。独立干净环境安装和鉴权后 Agent 的真实回答仍待验收。

2026-10-11 接入收尾：学习者实际验证凭证接入、本人可读与他人不可读、到期后退出聊天、状态及商品关键词列表查询、空列表、多订单确认后查询制度。修正登录检查请求头 Anthorization 拼写和 query_orders 漏注册。助手适配并补充离线检查，共14项通过，未再次调用付费API。独立干净环境安装、业务入口、人工工单和更广验收仍待完成。

2026-10-12 入口准备：学习者完成辅助函数及配置，助手源码复核并使用临时数据库检查本人可生成链接、他人和无效凭证不生成、非已发货返回场景限制，确认链接仅带订单号而没有凭证。共15项离线检查通过，pip check无冲突；不访问演示页、不调用付费模型。入口页和模型工具接入仍是下一步，不能用函数检查替代点击或对话验收。

2026-10-13 接入与工单：学习者实际报告页面本人/他人校验、签收后提供入口及未签收解释拒收两条分支通过。工单保存后端已写；助手复现未关联订单错误404，修正请求对象判断为order_id字段判断。20项离线检查通过，pip check无冲突。工单接口验证采用临时数据库，尚无Agent工单对话验收；下次接客户端与工具。页面模拟DOM七种状态检查只证明逻辑，不能代替真实浏览器界面验收。
