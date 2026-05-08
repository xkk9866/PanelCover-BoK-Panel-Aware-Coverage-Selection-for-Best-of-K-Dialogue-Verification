import sys, pathlib
p = pathlib.Path(sys.argv[1])
lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
out = []
for L in lines:
    if any(ch in L for ch in ["鑷潃", "鎴戣兘", "鎯虫", "宕╂", "璇风珛", "鍝勭棝", "鐨勭棝", "鎰熷", "鐜板湪", "鍜屾垜", "鎰挎剰", "棰勭儹"]):
        if "suicide" in L:
            out.append("(e.g. Chinese lexicon terms meaning ``suicide'', ``want to die'', ``collapse'').  On this")
            continue
        else:
            continue
    out.append(L)
p.write_text("\n".join(out), encoding="utf-8")
print("done")
