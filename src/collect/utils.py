"""通用工具和常量，用于各种爬虫模块。"""
from datetime import datetime, timedelta, timezone
from typing import Optional

# 默认的请求头，Referer可按来源覆盖
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def to_float(x: str) -> Optional[float]:
    """将字符串转换为浮点数，返回大于1的值否则None。"""
    try:
        x = (x or "").strip()
        if not x:
            return None
        v = float(x)
        return v if v > 1 else None
    except Exception:
        return None


def now_cn_date() -> str:
    """返回当前中国时区日期（YYYY-MM-DD）。"""
    dt = datetime.now(timezone.utc) + timedelta(hours=8)
    return dt.strftime("%Y-%m-%d")


def safe_read_html(html: str):
    """pandas.read_html包装，失败时返回空列表。"""
    import pandas as pd
    try:
        return pd.read_html(html)
    except Exception:
        return []


def decode_response(resp, default_encoding: str = "gbk") -> str:
    """Convert a :class:`requests.Response` to text in a reliable way.

    Historically the code relied on ``resp.apparent_encoding`` which in
    newer versions of ``requests``/``chardet`` started returning nonsense
    values (``"mac_greek"`` etc.) for the 500.com pages.  As a result the
    HTML sent to ``BeautifulSoup`` was decoded with the wrong codec and the
    output was full of Greek‑letter garbage, exactly what the user described
    after "upgrade".

    We avoid the whole detection dance by forcing a sane default (gbk/gb2312)
    which covers all of the Chinese sites we scrape.  The function still
    catches exceptions and will fall back to the ``resp.text`` property if the
    forced decode unexpectedly fails.
    """
    # try the default first; it's the simplest and avoids any broken
    # chardet/apparent_encoding heuristics
    try:
        return resp.content.decode(default_encoding, errors="replace")
    except Exception:
        enc = (resp.encoding or "").lower()
        try:
            return resp.content.decode(enc, errors="replace")
        except Exception:
            # last resort, let requests do its thing (may be wrong but at
            # least we won't crash)
            return resp.text
