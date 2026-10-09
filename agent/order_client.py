"""Read-only adapter for the local order API."""
from urllib.parse import quote

import httpx

from settings import ORDER_API_BASE_URL


def get_order(order_id: str) -> dict:
    response = httpx.get(
        f"{ORDER_API_BASE_URL}/orders/{quote(order_id, safe='')}", timeout=5.0
    )
    if response.status_code == 404:
        return {"error": "订单不存在", "order_id": order_id}
    response.raise_for_status()
    return response.json()
