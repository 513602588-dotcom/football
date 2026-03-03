import json
from datetime import datetime, timezone

from src.collect.wencai51 import fetch_wencai51

def main():
    with open("data/wencai51_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mt = cfg.get("mt","")
    api_url = cfg.get("api_url","")

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source": "m.wencai51.cn bonusCalculator",
            "mt": mt,
            "api_url": api_url,
        },
        "matches": []
    }

    try:
        if not api_url or api_url == "PASTE_API_URL_HERE":
            raise RuntimeError("api_url not set. Run probe first and paste it into data/wencai51_config.json")
        res = fetch_wencai51(api_url=api_url, mt=mt, ttl=600)
        payload["matches"] = res["matches"]
        payload["meta"]["count"] = len(payload["matches"])
    except Exception as e:
        payload["meta"]["error"] = str(e)
        payload["meta"]["count"] = 0

    with open("site/data/jczq.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("OK: site/data/jczq.json written, count=", payload["meta"]["count"])

if __name__ == "__main__":
    main()
