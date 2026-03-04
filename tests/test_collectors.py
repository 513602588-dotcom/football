import unittest
import pandas as pd

from src.collect import fetch_500_day, fetch_okooo_day


class CollectorSmokeTests(unittest.TestCase):
    def test_500_fetch(self):
        data = fetch_500_day("2026-03-04")
        self.assertIsInstance(data, list)
        if data:
            self.assertIn("home", data[0])
            self.assertIn("away", data[0])

            # after past upgrades the page started returning garbled text;
            # make sure at least one of the returned fields contains a
            # Chinese character (likely part of a league/team name).
            import re
            has_cn = any(re.search(r"[\u4e00-\u9fff]",
                                    str(item.get("home","")) + str(item.get("league","")))
                         for item in data)
            self.assertTrue(has_cn, "expected Chinese characters in 500 data")

    def test_okooo_fetch(self):
        # 获取当日数据，版本full
        df = fetch_okooo_day("2026-03-04", version="full")
        # 有时候目标网站未提供数据（返回 None），这也应被视为‘成功’。
        if df is None:
            return
        # 否则必须返回 DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        # 若有行，应包含标准列
        if len(df) > 0:
            self.assertTrue("home" in df.columns)
            self.assertTrue("away" in df.columns)


if __name__ == "__main__":
    unittest.main()
