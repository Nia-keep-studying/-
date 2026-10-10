from pwdlib import PasswordHash
from sqlmodel import Session

from .sqlmodel_server import User,Order,engine,create_db_and_tables

password_hasher = PasswordHash.recommended()

def seed_demo():
    create_db_and_tables()

    with Session(engine) as session:
        user1 = User(user_id="U1001",username="alice",password_hash=password_hasher.hash("demo-password"))
        user2 = User(user_id="U1002",username="bob",password_hash=password_hasher.hash("demo_password_2"))

        for user in [user1,user2]:
            if session.get(User,user.user_id) is None:
                session.add(user)
        session.commit()

        order1 = Order(order_id="A1001",user_id="U1001",product="电脑",status="已发货")
        order2 = Order(order_id="A1002",user_id="U1001",product="手机",status="已发货")
        order3 = Order(order_id="A2001",user_id="U1002",product="显示器",status="已发货")

        for order in [order1,order2,order3]:
            if session.get(Order,order.order_id) is None:
                session.add(order)
        session.commit()

if __name__=="__main__":
    seed_demo()