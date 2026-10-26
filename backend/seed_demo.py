import time

from pwdlib import PasswordHash
from sqlmodel import Session

from .sqlmodel_server import User,Order,engine,create_db_and_tables

password_hasher = PasswordHash.recommended()

def seed_demo():
    create_db_and_tables()
    now = int(time.time())
    seconds_per_day = 24 * 60 * 60

    with Session(engine) as session:
        user1 = User(user_id="U1001",username="alice",password_hash=password_hasher.hash("demo-password"))
        user2 = User(user_id="U1002",username="bob",password_hash=password_hasher.hash("demo_password_2"))

        for user in [user1,user2]:
            if session.get(User,user.user_id) is None:
                session.add(user)
        session.commit()

        # Receipt times are relative to the first insertion, not refreshed on every run.
        orders = [
            Order(order_id="A1001", user_id="U1001", product="电脑", status="已签收", received_at=now - 2 * seconds_per_day),
            Order(order_id="A1002", user_id="U1001", product="手机", status="已签收", received_at=now - 10 * seconds_per_day),
            Order(order_id="A1003", user_id="U1001", product="蓝牙耳机", status="待发货"),
            Order(order_id="A1004", user_id="U1001", product="机械键盘", status="已发货"),
            Order(order_id="A1005", user_id="U1001", product="平板电脑", status="已取消"),
            Order(order_id="A1006", user_id="U1001", product="无线鼠标", status="已签收", received_at=now - seconds_per_day),
            Order(order_id="A1007", user_id="U1001", product="手机保护壳", status="已签收", received_at=now - 5 * seconds_per_day),
            Order(order_id="A2001", user_id="U1002", product="显示器", status="已发货"),
            Order(order_id="A2002", user_id="U1002", product="路由器", status="已签收", received_at=now - 3 * seconds_per_day),
            Order(order_id="A2003", user_id="U1002", product="移动硬盘", status="已签收", received_at=now - 12 * seconds_per_day),
        ]

        for order in orders:
            if session.get(Order,order.order_id) is None:
                session.add(order)
        session.commit()

if __name__=="__main__":
    seed_demo()
