"""HTTP adapter for the local customer-support API."""
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import httpx

from settings import ORDER_API_BASE_URL


def _with_received_time_display(order: dict) -> dict:
    """Keep the numeric receipt time for checks and add a deterministic display value."""
    received_at = order.get("received_at")
    display = None
    if received_at is not None:
        try:
            received_time = datetime.fromtimestamp(
                received_at, tz=timezone(timedelta(hours=8))
            )
            display = received_time.strftime("%Y-%m-%d %H:%M:%S") + "（北京时间）"
        except (OverflowError, OSError, ValueError, TypeError):
            # Formatting must not hide an order whose receipt time needs investigation.
            display = None
    return {**order, "received_at_display": display}


def get_order(order_id: str,access_token:str) -> dict:
    response = httpx.get(
        f"{ORDER_API_BASE_URL}/orders/{quote(order_id, safe='')}",
        headers={"Authorization":f"Bearer {access_token}"},
        timeout=5.0
    )
    if response.status_code == 401:
        return {
            "code":"unauthenticated",
            "error":"未登录或登录已失效，请重新登陆",
        }

    if response.status_code == 404:
        return {"error": "订单不存在", "order_id": order_id}

    response.raise_for_status()
    return _with_received_time_display(response.json())

def get_order_list(access_token:str,status:str|None=None,product:str|None=None,product_keyword:str|None=None,offset:int=0,limit:int=10)->list[dict]|dict:
    params = {"offset":offset,"limit":limit}

    if status is not None:
        params["status"] = status

    if product is not None:
        params["product"] = product

    if product_keyword is not None:
        params["product_keyword"] = product_keyword

    response = httpx.get(
        f"{ORDER_API_BASE_URL}/orders",
        headers={"Authorization":f"Bearer {access_token}"},
        params=params,
        timeout=5
    )

    if response.status_code == 401:
        return {
            "code":"unauthenticated",
            "error":"未登录或登录已失效，请重新登陆",
        }

    response.raise_for_status()
    orders = response.json()
    if isinstance(orders, list):
        return [_with_received_time_display(order) for order in orders]
    return orders

def login(username:str,password:str)->dict:
    response = httpx.post(
        f"{ORDER_API_BASE_URL}/login",
        json={
            "username":username,
            "password":password,
        },
        timeout=5
    )
    response.raise_for_status()
    return response.json()

def is_logged_in(access_token:str)-> bool:
    response = httpx.get(
        f"{ORDER_API_BASE_URL}/me",
        headers={
            "Authorization":f"Bearer {access_token}"
        },
        timeout=5
    )

    if response.status_code==401:
        return False

    response.raise_for_status()
    return True

def create_human_ticket(
    access_token:str,
    summary:str,
    reason:str,
    conversation_id:str,
    idempotency_key:str,
    order_id:str|None = None,
    policy_reference:str|None = None,
) -> dict:
    json = {"summary":summary,"reason":reason,"conversation_id":conversation_id,"idempotency_key":idempotency_key}
    if order_id is not None:
        json["order_id"]=order_id
    if policy_reference is not None:
        json["policy_reference"]=policy_reference
    response = httpx.post(
        f"{ORDER_API_BASE_URL}/human-tickets",
        headers={
            "Authorization":f"Bearer {access_token}"
        },
        json=json,
        timeout=5
    )

    if response.status_code == 401:
        return {"code":"unauthenticated","error":"身份验证失败，请重新登录"}
    if response.status_code == 404:
        return {"error":"无法关联对应订单"}
    if response.status_code == 409:
        return {"error":"该编号已绑定其他工单，该操作编号对应的请求内容不一致，请核对原请求或已有工单。"}
    response.raise_for_status()
    return response.json()

def create_conversation(access_token:str):
    conversation = httpx.post(f"{ORDER_API_BASE_URL}/conversations",headers={"Authorization":f"Bearer {access_token}"},timeout=5)
    conversation.raise_for_status()
    conversation_dict = conversation.json()
    return conversation_dict["conversation_id"]

if __name__ == "__main__":
    from getpass import getpass

    username = input("用户名：").strip()
    password = getpass("密码：")

    login_result = login(username, password)
    access_token = login_result["access_token"]

    print("自己的订单：", get_order("A1001", access_token))
    print("其他用户的订单：", get_order("A2001", access_token))

    print(
        "已发货订单：",
        get_order_list(access_token, status="已发货"),
    )

    print(
        "手机订单：",
        get_order_list(access_token, product_keyword="手机"),
    )

    print(
        "无匹配订单：",
        get_order_list(access_token, product_keyword="不存在的商品"),
    )
