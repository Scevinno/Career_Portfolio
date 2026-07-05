"""Attribute each executable line of the portfolio project code to a Python library.

Method:
- Blank lines and comment-only lines are excluded; inline comments are stripped.
- Imports build a symbol->library map (e.g. pd -> pandas, train_test_split -> scikit-learn).
- Variables assigned from a tracked symbol inherit its library (one-hop provenance,
  e.g. model_data = pd.read_csv(...) makes model_data a pandas object).
- A line is attributed to the library of the first tracked symbol that makes a call
  or subscript on that line; if none is called, the first tracked symbol referenced.
- Lines referencing no tracked symbol count as core Python. Import lines count
  toward the imported library.
"""
import re
import sys
from collections import Counter

LIB_LABELS = {"pandas": "pandas", "sklearn": "scikit-learn", "numpy": "NumPy",
              "matplotlib": "Matplotlib", "seaborn": "seaborn"}

def measure(paths):
    symbols = {}   # name -> lib key
    counts = Counter()
    total = 0
    for path in paths:
        for raw in open(path, encoding="utf-8", errors="replace"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            code = re.split(r'(?<!["\'])#', line)[0].strip()
            if not code:
                continue
            total += 1

            m = re.match(r'import\s+([\w.]+)(?:\s+as\s+(\w+))?', code)
            if m:
                lib = m.group(1).split(".")[0]
                alias = m.group(2) or m.group(1).split(".")[0]
                symbols[alias] = lib
                counts[lib] += 1
                continue
            m = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', code)
            if m:
                lib = m.group(1).split(".")[0]
                for name in m.group(2).split(","):
                    name = name.strip().split(" as ")
                    symbols[name[-1].strip()] = lib
                counts[lib] += 1
                continue

            # find tracked symbols in order of appearance, prefer one being called/subscripted
            found = [(mm.start(), mm.group(1), bool(re.match(r'\s*[(\[.]', code[mm.end():])))
                     for mm in re.finditer(r'\b(\w+)\b', code) if (lambda g: g in symbols)(mm.group(1))]
            called = [f for f in found if f[2]]
            pick = (called or found or None)
            lib = symbols[pick[0][1]] if pick else "python"
            counts[lib] += 1

            # provenance: lhs vars inherit the line's library
            am = re.match(r'([\w\s,\[\]"\'%]+?)=[^=]', code)
            if am and lib != "python":
                for var in re.findall(r'\b(\w+)\b', am.group(1).split("[")[0]):
                    symbols.setdefault(var, lib)
    return counts, total

counts, total = measure(sys.argv[1:])
print(f"total executable lines: {total}")
for lib, n in counts.most_common():
    label = LIB_LABELS.get(lib, "core Python" if lib == "python" else lib)
    print(f"{label:14s} {n:4d}  {n/total*100:5.1f}%")
