"""Validation that follows marimo's semantics, looking at names in a cell's GLOBAL scope.
My first two attempts were both wrong.
  - looking at c.body missed anything inside try/if (it missed banks_msg)
  - looking with ast.walk also counted imports INSIDE functions (false alarm)
The right way is to walk the body but NOT descend into a nested def, class or lambda."""
import ast, re, sys
from pathlib import Path
from collections import defaultdict

def top_level_names(node):
    out=set()
    def walk(n, root=False):
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                if not root: continue
                out.add(ch.name); continue          # do not descend into its body
            if isinstance(ch,ast.Lambda): continue
            if isinstance(ch,(ast.Import,ast.ImportFrom)):
                for a in ch.names: out.add((a.asname or a.name).split('.')[0])
            tg=[]
            if isinstance(ch,ast.Assign): tg=ch.targets
            elif isinstance(ch,(ast.AnnAssign,ast.AugAssign)): tg=[ch.target]
            elif isinstance(ch,(ast.For,ast.AsyncFor)): tg=[ch.target]
            elif isinstance(ch,ast.With):
                tg=[it.optional_vars for it in ch.items if it.optional_vars]
            # Only DIRECT Name targets. `os.environ["X"] = v` is a Subscript
            # target. It contains Name('os') but it does NOT DEFINE `os`, it mutates it
            # Using ast.walk flagged os and sys as spurious collisions.
            def add_target(x):
                if isinstance(x,ast.Name): out.add(x.id)
                elif isinstance(x,(ast.Tuple,ast.List)):
                    for e in x.elts: add_target(e)
                elif isinstance(x,ast.Starred): add_target(x.value)
                # Subscript or Attribute means mutation, so it is skipped
            for x in tg: add_target(x)
            walk(ch)
    walk(node, root=True)
    return {n for n in out if not n.startswith('_')}

def renders(c):
    body=[s for s in c.body if not isinstance(s,ast.Return)]
    return bool(body) and isinstance(body[-1], ast.Expr)

p=Path(sys.argv[1] if len(sys.argv)>1 else 'asr_300h_marimo.py')
src=p.read_text(encoding='utf-8')
ast.parse(src)
cells=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)]

d=defaultdict(set)
for i,c in enumerate(cells):
    for n in top_level_names(c): d[n].add(i)
dup={k:sorted(v) for k,v in d.items() if len(v)>1}

prod=set(d)
unres=[(i,[a.arg for a in c.args.args if a.arg not in prod]) for i,c in enumerate(cells)]
unres=[u for u in unres if u[1]]

print(f"{p.name}: {len(cells)} cells, compiling")
print("multiple definitions:", dup or "none")
print("unresolved arguments:", unres or "none")

print("\nDISPLAY ORDER:")
cur=0; bad=[]
for i,c in enumerate(cells):
    if not renders(c): continue
    seg=ast.get_source_segment(src,c) or ""
    m=re.search(r'##\s+(\d+)\s*(?:\\u00b7|·)\s*([^\n"\\]{3,52})', seg)
    if m:
        n=int(m.group(1))
        flag = "" if n==cur+1 else f"   <-- OUT OF ORDER (expected {cur+1})"
        if flag: bad.append((i,n))
        cur=n
        print(f"  h{i:2d}  §{n:<2} {m.group(2).strip()[:46]}{flag}")
    else:
        h=re.search(r'#{2,3}\s+([^\n"\\0-9][^\n"\\]{2,50})', seg)
        if h: print(f"  h{i:2d}      ...{h.group(1).strip()[:46]}")
print("\nsection order:", "OK" if not bad else f"BROKEN: {bad}")
sys.exit(1 if (dup or unres or bad) else 0)
