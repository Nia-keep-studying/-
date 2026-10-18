from sqlmodel import SQLModel, Field,create_engine,Session,select
from settings import DATABASE_URL,LOGIN_SESSION_TTL_SECONDS,PROJECT_ROOT,CHECKPOINT_DATABASE_URL
from fastapi import FastAPI,HTTPException,Query,Response,Depends,Request
from typing import Literal
from contextlib import asynccontextmanager
from pwdlib import PasswordHash
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from uuid import uuid4
from langgraph.checkpoint.postgres import PostgresSaver

import secrets
import hashlib
import time


password_hasher = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)

OrderStatus = Literal["已发货","待发货","已取消"]

class User(SQLModel,table=True):
    __tablename__="users"

    user_id:str=Field(primary_key=True)
    username:str=Field(unique=True)
    password_hash:str

class LoginSession(SQLModel,table=True):
    __tablename__ = "login_session"

    token_hash:str = Field(primary_key=True)
    user_id:str = Field(foreign_key="users.user_id",index=True)
    expires_at:int

class LoginRequest(SQLModel):
    username:str
    password:str


class Order(SQLModel,table=True):
    __tablename__ = "orders"

    order_id:str = Field(primary_key=True)
    user_id:str=Field(foreign_key="users.user_id",index=True)
    product:str
    status:str

class HumanTicket(SQLModel,table=True):
    __tablename__ = "human_tickets"

    ticket_id:str=Field(primary_key=True)
    user_id:str=Field(foreign_key="users.user_id",index=True)
    order_id:str|None = Field(
        default=None,
        foreign_key="orders.order_id",
        index=True,
    )

    summary:str
    reason:str
    policy_reference:str|None = None
    conversation_id:str
    created_at:int
    status:str = "待处理"

class HumanTicketCreate(SQLModel):
    order_id:str|None = None
    summary:str=Field(min_length=1,max_length=2000)
    reason:str=Field(min_length=1,max_length=500)
    policy_reference:str|None=Field(
        default=None,
        max_length=2000,
    )
    conversation_id:str=Field(min_length=1,max_length=100)

class ChatConversation(SQLModel,table=True):
    __tablename__ = "chat_conversations"

    conversation_id:str = Field(primary_key=True)
    user_id:str = Field(foreign_key="users.user_id",index=True)
    created_at:int

class ChatRequest(SQLModel):
    question:str = Field(min_length=1,max_length=4000)
    conversation_id:str|None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

engine = create_engine(DATABASE_URL,echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(application: FastAPI):
    from agent.langchain_agent_demo import initialize_agent
    create_db_and_tables()

    with PostgresSaver.from_conn_string(
        CHECKPOINT_DATABASE_URL
    ) as checkpointer:
        checkpointer.setup()

        application.state.agent = initialize_agent(checkpointer=checkpointer)

        yield

app = FastAPI(lifespan=lifespan)

def get_session():
    with Session(engine) as session:
        yield session

def get_current_user(
        credentials:HTTPAuthorizationCredentials|None=Depends(bearer_scheme),
        session:Session=Depends(get_session)
)->User:
    auth_error = HTTPException(status_code=401,detail="未登录或登录已失效",headers={"WWW-Authenticate":"Bearer"})

    if credentials is None:
        raise auth_error

    token = credentials.credentials
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    login_session = session.get(LoginSession,token_hash)

    if login_session is None:
        raise auth_error

    if int(time.time()) >= login_session.expires_at:
        raise auth_error

    user = session.get(User,login_session.user_id)
    if user is None:
        raise auth_error

    return user

def get_orders(
        session:Session,
        user_id:str,
        status:OrderStatus|None = None,
        offset:int = 0,
        limit:int = 10,
        product:str|None = None,
        product_keyword:str|None = None,
        ):

    statement = select(Order).where(Order.user_id==user_id)
    if status is not None:
        statement = statement.where(Order.status == status)
    if product is not None:
        statement = statement.where(Order.product == product)
    if product_keyword is not None:
        statement = statement.where(Order.product.like(f"%{product_keyword}%"))
    statement = statement.offset(offset).limit(limit)
    orders = session.exec(statement).all()

    return orders

@app.post("/login")
def login(login_data:LoginRequest,session:Session=Depends(get_session),):
    statement = select(User).where(User.username==login_data.username)
    user = session.exec(statement).first()
    if user is None:
        raise HTTPException(status_code=401,detail="用户名或密码错误")

    password_ok = password_hasher.verify(login_data.password,user.password_hash,)
    if not password_ok:
        raise HTTPException(status_code=401,detail="用户名或密码错误")

    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = int(time.time())+LOGIN_SESSION_TTL_SECONDS

    login_session = LoginSession(
        token_hash=token_hash,
        user_id=user.user_id,
        expires_at=expires_at,
    )

    session.add(login_session)
    session.commit()

    return {
        "access_token":token,
        "token_type":"bearer",
        "expires_in":LOGIN_SESSION_TTL_SECONDS,
        "user_id":user.user_id,
        "username":user.username,
    }

@app.get("/me")
def read_current_user(current_user:User=Depends(get_current_user)):
    return {"user_id":current_user.user_id,"username":current_user.username}

@app.post("/logout",status_code=204)
def logout(
    current_user:User=Depends(get_current_user),
    credentials:HTTPAuthorizationCredentials=Depends(bearer_scheme),
    session:Session=Depends(get_session),
):
    token = credentials.credentials
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    login_session=session.get(LoginSession,token_hash)

    if login_session is not None:
        session.delete(login_session)
        session.commit()

    return Response(status_code=204)

@app.get("/orders")
def read_orders(
    current_user:User=Depends(get_current_user),
    status:OrderStatus|None = None,
    product:str|None = None,
    product_keyword:str|None = None,
    offset:int = Query(default=0,ge=0),
    limit:int = Query(default=10,ge=1,le=100),
    session:Session = Depends(get_session)
    ):
    orders = get_orders(user_id=current_user.user_id,status=status,offset=offset,limit=limit,session=session,product=product,product_keyword=product_keyword)
    return orders

@app.get("/orders/{order_id}")
def read_order(order_id:str,session:Session = Depends(get_session),current_user:User=Depends(get_current_user)):
    statement = select(Order).where(Order.user_id==current_user.user_id,Order.order_id==order_id)
    order = session.exec(statement).first()
    if order is None:
        raise HTTPException(status_code=404, detail="订单未找到")
    return order

@app.get("/demo/return-application",response_class=FileResponse,include_in_schema=False)
def return_application_page():
    return FileResponse(
        path=PROJECT_ROOT / "web" / "return_application.html",
        media_type="text/html",
        headers={"Cache-Control":"no-store"}
    )

@app.post("/human-tickets",status_code=201)
def create_human_ticket(
    ticket_data:HumanTicketCreate,
    current_user:User=Depends(get_current_user),
    session:Session=Depends(get_session),
):
    if ticket_data.order_id is not None:
        statement = select(Order).where(
            Order.order_id==ticket_data.order_id,
            Order.user_id==current_user.user_id,
        )
        order = session.exec(statement=statement).first()

        if order is None:
            raise HTTPException(
                status_code=404,
                detail="未找到可关联订单",
            )

    ticket = HumanTicket(
        ticket_id=str(uuid4()),
        user_id=current_user.user_id,
        order_id=ticket_data.order_id,
        summary=ticket_data.summary,
        reason=ticket_data.reason,
        policy_reference=ticket_data.policy_reference,
        conversation_id=ticket_data.conversation_id,
        created_at=int(time.time()),
        status="待处理"
    )

    session.add(ticket)
    session.commit()
    session.refresh(ticket)

    return {
        "ticket_id":ticket.ticket_id,
        "order_id":ticket.order_id,
        "summary":ticket.summary,
        "status":ticket.status,
        "message":"工单已保存，状态为待处理。当前为演示流程，尚无真实客服接入。"
    }

@app.post("/chat")
def chat(
    chat_data:ChatRequest,
    request:Request,
    current_user:User = Depends(get_current_user),
    credentials:HTTPAuthorizationCredentials = Depends(bearer_scheme),
    session:Session = Depends(get_session),
):
    from agent.langchain_agent_demo import AgentContext,run_chat_turn

    question = chat_data.question.strip()
    if not question:
        raise HTTPException(status_code=422,detail="问题不能为空")

    if chat_data.conversation_id is None:
        conversation_id = str(uuid4())

        conversation = ChatConversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            created_at=int(time.time())
        )
        session.add(conversation)
        session.commit()
    else:
        statement = select(ChatConversation).where(
            ChatConversation.conversation_id == chat_data.conversation_id,
            ChatConversation.user_id == current_user.user_id,
        )
        conversation = session.exec(statement).first()

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="会话不存在或不可访问",
            )

        conversation_id = conversation.conversation_id

    session.close()

    agent = request.app.state.agent

    context = AgentContext(
        access_token=credentials.credentials,
    )

    config = {
        "configurable":{"thread_id":conversation_id},
        "recursion_limit":10,
    }

    answer = run_chat_turn(
        agent=agent,
        question=question,
        config=config,
        context=context,
    )

    return {
        "conversation_id":conversation_id,
        "answer":answer,
    }

@app.get("/",response_class=FileResponse,include_in_schema=False)
def chat_page():
    return FileResponse(
        path=PROJECT_ROOT / "web" / "index.html",
        media_type="text/html",
        headers={"Cache-Control":"no-store"},
    )

@app.get("/conversations")
def list_conversations(
    current_user:User = Depends(get_current_user),
    session:Session = Depends(get_session),
    offset:int = Query(default=0,ge=0),
    limit:int = Query(default=20,ge=1,le=100),
):
    statement = (
        select(ChatConversation)
        .where(ChatConversation.user_id==current_user.user_id)
        .order_by(
            ChatConversation.created_at.desc(),
            ChatConversation.conversation_id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    conversations = session.exec(statement).all()

    return {
        "conversations":[
            {
                "conversation_id":conversation.conversation_id,
                "created_at":conversation.created_at,
            }
            for conversation in conversations
        ],
        "offset":offset,
        "limit":limit,
    }

@app.get("/conversations/{conversation_id}/messages")
def read_conversation_messages(
    conversation_id:str,
    request:Request,
    current_user:User = Depends(get_current_user),
    session:Session = Depends(get_session),
):
    statement = select(ChatConversation).where(
        ChatConversation.user_id == current_user.user_id,
        ChatConversation.conversation_id == conversation_id,
    )
    conversation = session.exec(statement).first()

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="会话不存在或不可访问",
        )

    session.close()

    agent = request.app.state.agent
    config = {
        "configurable":{"thread_id":conversation_id}
    }
    state = agent.get_state(config)

    visible_messages = []

    for message in state.values.get("messages",[]):
        if message.type == "human":
            role = "user"
        elif message.type == "ai" and not message.tool_calls:
            role = "assistant"
        else:
            continue

        content = str(message.text)
        if not content:
            continue

        visible_messages.append({
            "role":role,
            "content":content,
        })

    return {
        "conversation_id":conversation_id,
        "messages":visible_messages,
    }
