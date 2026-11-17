"""Best (minimum) ns/vector per kernel and size across interleaved runs of run_kernels.sh."""
import re
import sys

best = {}
order = []
size = None
for line in open(sys.argv[1]):
    m = re.match(r"## iter \d+ n=(\d+)", line)
    if m:
        size = m.group(1)
        continue
    m = re.match(r"\| (C|Velt\(\w+\)) (.+?) \| ([\d.]+) \|", line)
    if m and size:
        key = (size, m.group(1), m.group(2))
        v = float(m.group(3))
        if key not in best:
            order.append(key)
            best[key] = v
        best[key] = min(best[key], v)

for s in sorted({k[0] for k in order}, key=int):
    print(f"\n### n={s}\n\n| impl | kernel | best ns/vector |\n|---|---|---|")
    for k in order:
        if k[0] == s:
            print(f"| {k[1]} | {k[2]} | {best[k]:.2f} |")
