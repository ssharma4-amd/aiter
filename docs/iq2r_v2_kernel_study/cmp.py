import re, sys
def load(f):
    out = {}
    for line in open(f):
        m = re.match(r"\S+ I=(\d+) M=\s*(\d+) (\S+) \| (.*) \| sum\s+([\d.]+) full\s+([\d.]+)", line)
        if not m: continue
        st = dict(re.findall(r"(\w+)\s+([\d.]+)", m.group(4)))
        out[int(m.group(2))] = (m.group(3), {k: float(v) for k, v in st.items()}, float(m.group(6)), "MISMATCH" in line)
    return out
if __name__ == "__main__":
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(f"{'M':>5} {'config':28} {'gateA':>7} {'gateB':>7} {'downA':>7} {'downB':>7} {'fullA':>7} {'fullB':>7} {'B/A':>6}")
    for m in sorted(a):
        if m not in b: continue
        ca, sa, fa, xa = a[m]; cb, sb, fb, xb = b[m]
        print(f"{m:5d} {ca + ('' if ca == cb else '|' + cb):28} {sa.get('gate',0):7.2f} {sb.get('gate',0):7.2f} {sa.get('down',0):7.2f} {sb.get('down',0):7.2f} {fa:7.2f} {fb:7.2f} {(fb/fa if fa else (sum(sb.values())/max(sum(sa.values()),1e-9))):6.3f}{' X' if xa or xb else ''}")
