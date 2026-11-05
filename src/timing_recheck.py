"""级联推理计时复核（审稿一致性：tab_timing 脚注 492+196 ms 与端到端 535 ms 不符）。

根因：timing_table.bench_inference 的组件耗时是在**独立的紧凑循环**里摊销测的
（stage-1 = 50 次 batch_argmax 连跑取均值；stage-2 = 对全零 crop 连跑 50 次），
而端到端是逐切片 perf_counter 计时的完整 cascade_predict（含 argmax/质心/裁剪/
softmax/贴回 + 空预测 tile fallback 检查）。两组测量条件不同（共享 CPU 上负载
随时间漂移），故组件之和（688 ms）不要求等于端到端均值（535 ms）。

本脚本在**同一个逐切片循环**内分解计时：stage1 前向+argmax / 质心+裁剪 /
stage2 前向+softmax / 贴回（+ 另测一遍完整 cascade_predict 作对照），
组件之和按构造与端到端自洽。CPU 运行，与 timing.json 相同的 50 切片与 ckpt。

产出：results/v2/timing_recheck.json
用法：python src/timing_recheck.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import torch
    import cascade
    import common
    from timing_table import load_bench_images

    device = "cpu"
    crop_size = 256
    images = load_bench_images("data/cache", "data/CARE", 50)
    m1 = common.load_model("runs/s1_100/best.pt", device)
    m2 = common.load_model("runs/sup_100/best.pt", device)
    m1.eval()
    m2.eval()
    x_all = torch.from_numpy(images).to(device)
    n = x_all.shape[0]

    rec = {"s1": [], "centroid_crop": [], "s2_fwd": [], "paste": [],
           "total_decomposed": [], "total_cascade_predict": []}
    n_empty = 0
    with torch.no_grad():
        for i in range(min(3, n)):  # warmup
            cascade.cascade_predict(m1, m2, x_all[i:i + 1], crop_size,
                                    batch_size=1)
        for i in range(n):
            x = x_all[i:i + 1]
            t0 = time.perf_counter()
            pred1 = cascade.batch_argmax(m1, x, batch_size=1)
            t1 = time.perf_counter()
            empty = not bool((pred1 > 0).any())
            n_empty += int(empty)
            offsets = cascade.crop_offsets(
                cascade.fg_centers(pred1), crop_size, 512, 512)
            crops, _ = cascade.crop_batch(x, offsets, crop_size)
            t2 = time.perf_counter()
            logits = m2(crops)
            probs = torch.softmax(logits.float(), dim=1)
            conf, pred = probs.max(1)
            t3 = time.perf_counter()
            canvas = cascade.paste_batch(pred.to(torch.uint8), offsets,
                                         crop_size, 512, 512)
            _ = cascade.paste_batch(conf, offsets, crop_size, 512, 512)
            t4 = time.perf_counter()
            rec["s1"].append((t1 - t0) * 1000)
            rec["centroid_crop"].append((t2 - t1) * 1000)
            rec["s2_fwd"].append((t3 - t2) * 1000)
            rec["paste"].append((t4 - t3) * 1000)
            rec["total_decomposed"].append((t4 - t0) * 1000)
            t5 = time.perf_counter()
            cascade.cascade_predict(m1, m2, x, crop_size, batch_size=1)
            rec["total_cascade_predict"].append((time.perf_counter() - t5)
                                                * 1000)

    def stat(v):
        a = np.asarray(v)
        return {"mean": float(a.mean()), "sd": float(a.std()),
                "median": float(np.median(a))}

    out = {
        "device": "cpu",
        "n_slices": n,
        "crop_size": crop_size,
        "ckpts": {"stage1": "runs/s1_100/best.pt",
                  "stage2": "runs/sup_100/best.pt"},
        "note": "组件与端到端在同一逐切片循环内计时（perf_counter），"
                "排除测量条件漂移；total_cascade_predict 为完整 "
                "cascade_predict（含空预测检查与 tile fallback 分支）。"
                "分解路径未触发 tile fallback 时与 cascade_predict 等价。",
        "n_stage1_empty": n_empty,
        "per_stage_ms": {k: stat(v) for k, v in rec.items()},
    }
    parts = (np.mean(rec["s1"]) + np.mean(rec["centroid_crop"])
             + np.mean(rec["s2_fwd"]) + np.mean(rec["paste"]))
    out["sum_of_parts_ms"] = float(parts)
    out["overhead_vs_cascade_predict_ms"] = float(
        np.mean(rec["total_cascade_predict"]) - parts)
    os.makedirs("results/v2", exist_ok=True)
    with open("results/v2/timing_recheck.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (round(v["mean"], 1), round(v["sd"], 1))
                      for k, v in out["per_stage_ms"].items()},
                     indent=1))
    print(f"sum_of_parts={parts:.1f} ms; "
          f"cascade_predict={np.mean(rec['total_cascade_predict']):.1f} ms; "
          f"stage1_empty={n_empty}/{n}")
    print("已保存 results/v2/timing_recheck.json")


if __name__ == "__main__":
    main()
