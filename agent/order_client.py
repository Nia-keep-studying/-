"""Read-only adapter for the local order API."""
from urllib.parse import quote

import httpx

from settings import ORDER_API_BASE_URL


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
    return response.json()

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
    return response.json()

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