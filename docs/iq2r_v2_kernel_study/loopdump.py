"""Print the innermost loop of the first kernel matching SUB. usage: loopdump.py FILE.s SUB"""
import re, sys
txt = open(sys.argv[1]).read()
m = next(re.finditer(r"\n(_ZN5aiter\S*" + re.escape(sys.argv[2]) + r"\S*):[^\n]*\n(.*?)s_endpgm", txt, re.S))
body = m.group(2).split("\n")
s = next(i for i, l in enumerate(body) if re.match(r"\.LBB\S+:", l) and "Inner Loop Header" in body[i + 1])
lab = body[s].split(":")[0]
e = next(i for i in range(s + 1, len(body)) if re.search(r"s_cbranch\S* " + re.escape(lab) + r"\b", body[i]))
for l in body[s:e + 1]:
    l = l.split(";")[0].rstrip()
    if l.strip(): print(l)
