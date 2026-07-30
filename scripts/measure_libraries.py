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

With --unique, an executable line that has already been counted anywhere in the
corpus is skipped, so each DISTINCT line counts once. A project built as a
series of variants (the CNN ladder: one architecture, five scripts, one change
each) would otherwise multiply a single body of work by five and swamp every
other project. This is the mode the published figure uses.

Usage:  python scripts/measure_libraries.py [--unique] <file> [<file> ...]
"""
import re
import sys
from collections import Counter

LIB_LABELS = {"pandas": "pandas", "sklearn": "scikit-learn", "numpy": "NumPy",
              "matplotlib": "Matplotlib", "seaborn": "seaborn",
              # Keras and Keras Tuner report as TensorFlow — one library on the bar.
              "tensorflow": "TensorFlow", "keras": "TensorFlow",
              "keras_tuner": "TensorFlow",
              "scipy": "SciPy", "causalimpact": "CausalImpact"}

def measure(paths, unique=False):
    symbols = {}   # name -> lib key
    counts = Counter()
    seen = set()   # distinct executable lines, when unique is on
    total = 0
    for path in paths:
        for raw in open(path, encoding="utf-8", errors="replace"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            code = re.split(r'(?<!["\'])#', line)[0].strip()
            if not code:
                continue
            if unique:
                if code in seen:
                    continue   # symbols map already holds this line's bindings
                seen.add(code)
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

args = sys.argv[1:]
unique = "--unique" in args
counts, total = measure([a for a in args if a != "--unique"], unique=unique)
print(f"total executable lines: {total}" + ("  (distinct only)" if unique else ""))
# Fold by display label first — several import keys can share one label
# (tensorflow, keras and keras_tuner all report as TensorFlow).
by_label = Counter()
for lib, n in counts.items():
    by_label[LIB_LABELS.get(lib, "core Python" if lib == "python" else lib)] += n
for label, n in by_label.most_common():
    print(f"{label:14s} {n:4d}  {n/total*100:5.1f}%")
