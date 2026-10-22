# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
一键跑完整流程 (run_all.py)

执行顺序（顺序执行，避免两套回测同时占用内存导致系统换页）:
  1. 重建数据面板（纳入补抓完整的 ST 快照）
  2. 计算 24 个因子（与股票池无关，两个口径共用）
  3. LSY 股票池全流程回测（主口径）
  4. 全 A 股票池全流程回测（稳健性对照）
"""
import os, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

STEPS = [
    ("重建数据面板（含完整 ST 快照）", [PY, "build_panel.py"]),
    ("计算 24 个因子", [PY, "ashare_factor_strategy.py", "--stage", "factors",
                    "--force-factors"]),
    ("LSY 股票池 · 全流程回测（主口径）", [PY, "ashare_factor_strategy.py",
                                   "--stage", "all"]),
    ("全 A 股票池 · 全流程回测（稳健性对照）", [PY, "ashare_factor_strategy.py",
                                      "--stage", "all", "--lsy", "0"]),
]

t_all = time.time()
for i, (label, cmd) in enumerate(STEPS, 1):
    print("\n" + "=" * 78)
    print(f"  [{i}/{len(STEPS)}] {label}")
    print(f"  {' '.join(cmd[1:])}")
    print("=" * 78, flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=BASE, env=ENV)
    print(f"\n  -> 退出码 {p.returncode}, 用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    if p.returncode != 0:
        print(f"  !! [{label}] 失败，中止后续步骤", flush=True)
        break

print(f"\n全部完成，总用时 {(time.time()-t_all)/60:.1f} 分钟")
print("产出: output/report_LSY.md, output/report_FULL.md, output/*.png, output/summary_*.json")
