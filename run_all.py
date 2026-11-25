"""Chạy toàn bộ pipeline 8 giai đoạn.

    python run_all.py                 # chạy tất cả
    python run_all.py --from 4        # chạy từ giai đoạn 4
    python run_all.py --only 6        # chỉ chạy giai đoạn 6
    python run_all.py --report        # sinh lại báo cáo Word từ results/
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, log  # noqa: E402

STAGES = [
    ("0", "Thiết kế bài toán & sơ đồ", "s0_design"),
    ("1", "Dữ liệu, checksum, hồ sơ dữ liệu", "s1_data"),
    ("2", "Đặc trưng theo thời gian & chia tách", "s2_features"),
    ("3", "Dự báo, hiệu chỉnh, bất định", "s3_forecast"),
    ("4", "Nhân quả & uplift", "s4_causal"),
    ("5", "Predict-Then-Optimize & SPO+", "s5_spo"),
    ("6", "Tối ưu MILP", "s6_milp"),
    ("7", "Action Router", "s7_router"),
    ("8", "MLOps & giám sát", "s8_mlops"),
    ("9", "Phân tích mở rộng & kiểm tra độ vững", "s9_extended"),
    ("10", "Bằng chứng bổ sung: lực thống kê và mô phỏng có đối chứng", "s10_evidence"),
    ("11", "EconML, SHAP và học chính sách", "s11_causal_xai"),
    ("12", "MLflow, Evidently và kiểm thử FastAPI", "s12_mlops_tools"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    timings = {}
    tpath = cfg["paths"]["results"] / "stage_timings.json"
    if tpath.exists():
        timings = json.loads(tpath.read_text(encoding="utf-8"))
    if not args.report:
        for sid, name, mod in STAGES:
            if args.only is not None and sid != args.only:
                continue
            if args.only is None and int(sid) < int(args.start):
                continue
            log.info(f"===== Giai đoạn {sid}: {name} =====")
            t0 = time.perf_counter()
            m = __import__(mod)
            if hasattr(m, "main"):
                m.main()
            elif hasattr(m, "run"):
                m.run(cfg)
            else:
                m.build(cfg)
            timings[f"{sid}. {name}"] = round(time.perf_counter() - t0, 1)
            tpath.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report or args.only is None:
        rp = Path(__file__).parent / "report" / "build_report.py"
        if rp.exists():
            sys.path.insert(0, str(rp.parent))
            import build_report
            build_report.main()


if __name__ == "__main__":
    main()
