"""Chụp ảnh các trang của dashboard Streamlit để đưa vào báo cáo.
Chạy: python dashboard/capture_dashboard.py  (tự khởi động Streamlit ở cổng 8599)."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
p = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--server.headless", "true",
                      "--server.port", "8599", "--browser.gatherUsageStats", "false"], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(8)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
        for i in range(1, 6):
            pg.goto(f"http://localhost:8599/?page={i}", wait_until="networkidle")
            pg.wait_for_timeout(6000)
            pg.screenshot(path=str(OUT / f"fx3{i}_dashboard_page{i}.png"), full_page=True)
            print("ok", i)
        b.close()
finally:
    p.terminate()
