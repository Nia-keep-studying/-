from .order_client import get_order
from urllib.parse import urlencode
from settings import RETURN_APPLICATION_URL

def get_return_entry(order_id: str, access_token: str) -> dict:
    order = get_order(order_id=order_id,access_token=access_token)

    if "error" in order:
        return order

    if order["status"] != "已发货":
        return {
            "code":"unsupported_status",
            "error":"当前退货入口暂只支持已发货订单，请按对应状态的制度处理",
            "order_id":order["order_id"],
            "status":order["status"],
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
        "entry_name": "退货申请入口",
        "entry_url": entry_url,
        "is_demo": True,
        "message": (
            "这是演示退货申请入口，请进入页面继续办理。"
            "提供入口不表示申请已提交或退款已批准。"
        ),
    }