"""独立的爬虫运行脚本

此脚本按顺序运行500网和澳客历史数据爬虫并将结果写入site/data目录，
用于在不启动完整管道的情况下更新原始数据。
"""

from src.collect import export_500, export_okooo, now_cn_date


def main():
    print("开始运行外部爬虫...")
    try:
        export_500(days=1)
        today = now_cn_date()
        export_okooo(start_date=today, days=3, version="full")
        print("爬虫执行完毕，请查看 site/data 下的 JSON 文件。")
    except Exception as e:
        print("爬虫运行时发生异常:", e)


if __name__ == "__main__":
    main()
