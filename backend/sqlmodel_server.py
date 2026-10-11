from sqlmodel import SQLModel, Field,create_engine,Session,select
from settings import DATABASE_PATH,LOGIN_SESSION_TTL_SECONDS
from fastapi import FastAPI,HTTPException,Query,Response,Depends
from typing import Literal
from contextlib import asynccontextmanager
from pwdlib import PasswordHash
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials

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

database_file = DATABASE_PATH
database_url = f"sqlite:///{database_file.as_posix()}"
engine = create_engine(database_url, echo=False, connect_args={"check_same_thread": False})

def create_db_and_tables():
    database_file.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(application: FastAPI):
    create_db_and_tables()
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
