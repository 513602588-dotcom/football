import sys
from playwright.sync_api import sync_playwright

def main():
    url = sys.argv[1]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def on_response(resp):
            ct = resp.headers.get("content-type", "")
            if "json" in ct:
                print("JSON:", resp.status, resp.url)

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(5000)
        browser.close()

if __name__ == "__main__":
    main()
