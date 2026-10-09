from sqlmodel import SQLModel, Field,create_engine,Session,select
from settings import DATABASE_PATH
from fastapi import FastAPI,HTTPException,Query,Response,Depends
from typing import Literal
from contextlib import asynccontextmanager

OrderStatus = Literal["已发货","待发货","已取消"]

class OrderStatusUpdate(SQLModel):
    status:OrderStatus

class OrderUpdate(SQLModel):
    product:str
    status:OrderStatus


class OrderCreate(SQLModel):
    order_id:str
    product:str
    status:OrderStatus

class Order(SQLModel,table=True):
    __tablename__ = "orders"

    order_id:str = Field(primary_key=True)
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

def can_status_change(old_status:str,new_status:OrderStatus)->bool:
    return not(old_status == "已取消" and new_status != "已取消")

def add_order(order_data: OrderCreate,session:Session):
    order = Order(**order_data.model_dump())


    session.add(order)
    session.commit()
    session.refresh(order)
    return order

def get_orders(session:Session,status:OrderStatus|None = None,offset:int = 0,limit:int = 10,product:str|None = None,product_keyword:str|None = None):

    statement = select(Order)
    if status is not None:
        statement = statement.where(Order.status == status)
    if product is not None:
        statement = statement.where(Order.product == product)
    if product_keyword is not None:
        statement = statement.where(Order.product.like(f"%{product_keyword}%"))
    statement = statement.offset(offset).limit(limit)
    orders = session.exec(statement).all()

    return orders

def update_order_status(order_id:str,new_status:OrderStatus,session:Session):

    order = session.get(Order,order_id)
    if order is None:
        return None
    if can_status_change(old_status=order.status,new_status=new_status) == False:
        return False
    order.status = new_status
    session.add(order)
    session.commit()
    session.refresh(order)
    return order


def get_order_by_id(order_id:str,session:Session):

    statement = select(Order).where(Order.order_id == order_id)
    order = session.exec(statement).first()
    return order

def delete_order(order_id:str,session:Session) -> bool:

    order = session.get(Order, order_id)
    if order is None:
        return False
    session.delete(order)
    session.commit()
    return True

def order_update(order_id:str,update_data:OrderUpdate,session:Session):
    order = session.get(Order,order_id)
    if order is None:
        return None
    if can_status_change(old_status=order.status,new_status=update_data.status) == False:
        return False
    order.product = update_data.product
    order.status = update_data.status

    session.commit()
    session.refresh(order)
    return order

@app.get("/orders")
def read_orders(
    status:OrderStatus|None = None,
    product:str|None = None,
    product_keyword:str|None = None,
    offset:int = Query(default=0,ge=0),
    limit:int = Query(default=10,ge=1,le=100),
    session:Session = Depends(get_session)
    ):
    orders = get_orders(status=status,offset=offset,limit=limit,session=session,product=product,product_keyword=product_keyword)
    return orders

@app.get("/orders/{order_id}")
def read_order(order_id:str,session:Session = Depends(get_session)):
    order = get_order_by_id(order_id,session=session)
    if order is None:
        raise HTTPException(status_code=404, detail="订单未找到")
    return order

@app.post("/orders",status_code = 201)
def create_order(new_order:OrderCreate,session:Session = Depends(get_session)):
    existing_order = get_order_by_id(new_order.order_id,session=session)
    if existing_order is not None:
        raise HTTPException(status_code=409, detail="订单ID已存在")

    order = add_order(new_order,session=session)
    return order

@app.patch("/orders/{order_id}/status")
def change_order_status(order_id:str,update_data:OrderStatusUpdate,session:Session = Depends(get_session)):

    order = update_order_status(order_id,new_status=update_data.status,session=session)
    if order is None:
        raise HTTPException(status_code=404,detail="订单未找到")
    if order == False:
        raise HTTPException(status_code=403,detail="无法修改已取消的订单状态")
    return order

@app.delete("/orders/{order_id}",status_code=204)
def remove_order(order_id:str,session:Session = Depends(get_session)):
    deleted = delete_order(order_id,session=session)
    if deleted is False:
        raise HTTPException(status_code=404,detail="订单未找到")
    return Response(status_code=204)

@app.put("/orders/{order_id}")
def update_order(order_id:str,update_data:OrderUpdate,session:Session = Depends(get_session)):
    order = session.get(Order,order_id)
    if order is None:
        raise HTTPException(status_code=404,detail="没找到订单")
    updated = order_update(order_id=order_id,update_data=update_data,session=session)
    if updated is None:
        raise HTTPException(status_code=404,detail="没有找到对应订单")
    elif updated == False:
        raise HTTPException(status_code=403,detail="无法更改已经取消的订单")

    return updated
