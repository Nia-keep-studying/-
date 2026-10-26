from .order_client import get_order
from urllib.parse import urlencode
from settings import RETURN_APPLICATION_URL
import time

def get_return_entry(order_id: str,access_token: str) -> dict:
    order = get_order(order_id=order_id,access_token=access_token)

    if "error" in order:
        return order

    if order["status"] != "已签收":
        return {
            "code":"unsupported_status",
            "error":"当前退货入口暂只支持已签收订单，请按对应状态的制度处理",
            "order_id":order["order_id"],
            "status":order["status"],
        }

    received_at = order.get("received_at")
    if received_at is None:
        return{
            "code":"no_received_time",
            "error":"缺少签收时间，暂时无法确认是否符合七天期限",
            "order_id":order["order_id"],
            "status":order["status"],
        }

    elapsed_seconds = int(time.time()) - received_at

    if elapsed_seconds <0:
        return {
            "code":"received_time_error",
            "error":"签收时间有误，签收时间在未来",
            "order_id":order["order_id"],
            "status":order["status"],
            "received_at":order["received_at"],
            "received_at_display":order.get("received_at_display"),
        }
    elif elapsed_seconds > 7*24*60*60:
        return {
            "code":"out_of_return_time",
            "details":"距签收已超过七天，不符合本演示入口的时间条件，其他售后请按制度进一步核实。",
            "order_id":order["order_id"],
            "status":order["status"],
            "received_at":order["received_at"],
            "received_at_display":order.get("received_at_display"),
        }

    query = urlencode(
        {
            "order_id":order["order_id"],
        }
    )
    entry_url = f"{RETURN_APPLICATION_URL}?{query}"

    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "received_at_display": order.get("received_at_display"),
        "entry_name": "退货申请入口",
        "entry_url": entry_url,
        "is_demo": True,
        "message": (
            "这是演示退货申请入口，请进入页面继续办理。"
            "提供入口不表示申请已提交或退款已批准。"
        ),
    }
