import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf

load_dotenv()

rag = RagSummarizeService()

QWEATHER_TIMEOUT = 15
IP_API_URL = "http://ip-api.com/json/"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _get_qweather_base_url() -> str:
    """读取和风天气控制台分配的 API Host。"""
    host = os.getenv("QWEATHER_API_HOST", "").strip().rstrip("/")
    if not host:
        raise RuntimeError(
            "未配置 QWEATHER_API_HOST，请从和风天气控制台复制 API Host"
        )
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return host


def _qweather_get(path: str, params: dict) -> dict:
    """请求和风天气接口，并统一检查 HTTP 与业务状态码。"""
    api_key = os.getenv("QWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 QWEATHER_API_KEY")

    url = f"{_get_qweather_base_url()}{path}"
    response = requests.get(
        url,
        params={**params, "key": api_key},
        timeout=QWEATHER_TIMEOUT,
    )
    if not response.ok:
        try:
            error = response.json().get("error", {})
            detail = error.get("title") or error.get("detail")
        except ValueError:
            detail = None
        message = detail or f"HTTP {response.status_code}"
        raise RuntimeError(f"和风天气请求失败：{message}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("和风天气返回了无法解析的数据") from exc

    if str(data.get("code")) != "200":
        raise RuntimeError(f"和风天气返回错误码：{data.get('code', '未知')}")
    return data


def _geocode_city(city: str) -> dict:
    """GeoAPI 不可用时，用免费地理编码服务解析城市坐标。"""
    city = city.strip()
    # 免费地理编码对“珠海”和“珠海市”的结果可能不同，两个名称都尝试
    query_names = {city}
    if city.endswith("市"):
        query_names.add(city[:-1])
    else:
        query_names.add(f"{city}市")

    places = {}
    for query_name in query_names:
        response = requests.get(
            GEOCODING_API_URL,
            params={
                "name": query_name,
                "count": 10,
                "language": "zh",
                "format": "json",
            },
            timeout=10,
        )
        response.raise_for_status()
        for place in response.json().get("results") or []:
            key = (place.get("name"), place.get("latitude"), place.get("longitude"))
            places[key] = place

    if not places:
        raise RuntimeError(f"未找到城市“{city}”")

    # 优先中国境内、行政级别较高且人口较多的同名地点
    feature_scores = {"PPLC": 40, "PPLA": 30, "PPLA2": 25, "PPLA3": 20, "PPLA4": 10}
    place = max(
        places.values(),
        key=lambda item: (
            item.get("country_code") == "CN",
            feature_scores.get(item.get("feature_code"), 0),
            item.get("population") or 0,
        ),
    )
    display_name = place.get("name", city)
    if place.get("admin1"):
        display_name = f"{display_name}（{place['admin1']}）"
    return {
        "name": display_name,
        "longitude": place["longitude"],
        "latitude": place["latitude"],
    }


@tool(description="从向量存储里检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    city = city.strip()
    if not city:
        return "天气查询失败：城市名称不能为空"

    try:
        try:
            geo_data = _qweather_get(
                "/geo/v2/city/lookup",
                {"location": city, "lang": "zh"},
            )
            locations = geo_data.get("location", [])
            if not locations:
                raise RuntimeError(f"未找到城市“{city}”")
            location = locations[0]
            location_query = location["id"]
            display_name = location.get("name", city)
        except (RuntimeError, KeyError):
            # 和风天气 GeoAPI 可能受账号权限限制，改用免费地理编码
            place = _geocode_city(city)
            location_query = f"{place['longitude']},{place['latitude']}"
            display_name = place["name"]

        weather_data = _qweather_get(
            "/v7/weather/now",
            {"location": location_query, "lang": "zh"},
        )
        now = weather_data.get("now", {})
        if not now:
            return f"天气查询失败：未获取到“{city}”的实时天气"

        return (
            f"{display_name}：{now.get('text', '未知')}，"
            f"气温 {now.get('temp', '--')}℃，"
            f"体感 {now.get('feelsLike', '--')}℃，"
            f"湿度 {now.get('humidity', '--')}%，"
            f"{now.get('windDir', '')}{now.get('windScale', '--')}级，"
            f"观测时间 {now.get('obsTime', '未知')}。"
        )
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
        return f"天气查询失败：{exc}"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    try:
        response = requests.get(
            IP_API_URL,
            params={
                "lang": "zh-CN",
                "fields": "status,message,country,regionName,city,query",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return "未知"

    if data.get("status") != "success":
        return "未知"
    return str(
        data.get("city")
        or data.get("regionName")
        or data.get("country")
        or "未知"
    )


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    # 接入真实登录系统后，可把当前用户 ID 写入 AGENT_USER_ID
    return os.getenv("AGENT_USER_ID", "user_001")


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")


@tool(
    description=(
        "获取指定用户在指定月份的家庭设备使用记录，以 JSON 字符串返回；"
        "支持净水器、空调和除湿机，未检索到时返回空数组"
    )
)
def fetch_external_data(user_id: str, month: str) -> str:
    """从 external_data.json 读取并按用户、月份筛选设备记录。"""
    project_root = Path(__file__).resolve().parents[2]
    data_path = Path(agent_conf.get("external_data_path", "data/external_data.json"))
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        return "[]"

    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "[]"

    records = data.get("records", []) if isinstance(data, dict) else []
    if not isinstance(records, list):
        return "[]"

    matched = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("user_id") == user_id
        and record.get("month") == month
    ]
    return json.dumps(matched, ensure_ascii=False)


@tool(
    description=(
        "无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，"
        "为后续提示词切换提供上下文信息"
    )
)
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"


# 兼容旧名称，避免已有调用方失效
file_context_for_report = fill_context_for_report
