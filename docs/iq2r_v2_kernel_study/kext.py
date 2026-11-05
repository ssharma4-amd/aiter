"""usage: kext.py FILE.s PATTERN OUT -> body of the first kernel matching PATTERN"""
import re, sys
f, pat, out = sys.argv[1:4]
name, body, done = None, [], False
for line in open(f):
    m = re.match(r"^(_Z\S+):\s*(;.*)?$", line)
    if m and not done:
        name, body = m.group(1), []
    elif name is not None and not done:
        body.append(line)
        if line.strip().startswith("s_endpgm"):
            if re.search(pat, name): done = True; break
            name = None
open(out, "w").write("".join(body))
