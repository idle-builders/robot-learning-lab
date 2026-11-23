"""Records what a generator process reads and writes, for the generator cache (robot/body/bom/gencache.py, make.py).

On PYTHONPATH only while make.py runs a generator ($GENCACHE_LOG set); does nothing otherwise. Every Python process of the
run (the generator and its Python subprocesses) writes $GENCACHE_LOG/<pid>.json at exit with repository-relative paths:
- reads: files opened for reading (Python's `open` audit event: json, numpy, trimesh, every module source), plus the files
  handed to native readers whose reads Python never sees: build123d import_step / import_brep / import_stl, OpenCascade's
  STEPControl_Reader / STEPCAFControl_Reader.ReadFile and BRepTools.Read_s, gmsh.open / gmsh.merge;
- modules: the source file of every module under the repository loaded at exit (a module loaded from __pycache__ never opens
  its .py, so the source is recorded from sys.modules);
- dirs: folders listed (os.listdir / os.scandir / glob), whose contents can change a result; the body definitions' search
  (robot/body/bom/definitions.py discover, an os.walk over all of robot/) is recorded as one key instead,
  "glob:robot/**/*.bom.yaml", the set of definition files: its folder listings made every untracked or ignored leftover
  folder under robot/ an input, so a fresh worktree reused nothing a main checkout with such folders had made (measured
  2026-11-23: roll-flip-cad and the 8 generators after it reran, about 10 min, in a worktree of the commit main had built);
- writes: files opened for writing or renamed into place (used to name undeclared outputs).
STEP files written through OpenCascade (STEPControl_Writer / STEPCAFControl_Writer.Write, which build123d export_step uses)
get a fixed header time (2000-01-01T00:00:00) right after the write, before the builder hashes them: OpenCascade stamps the
write time into FILE_NAME, the manifests record each STEP's raw sha256, and so every rerun that wrote the same geometry
changed the manifest and everything after it (measured 2026-11-22: 27 manifest rows differed between two passes only by
these hashes). The geometry is untouched; STEPs written outside make keep OpenCascade's stamp.
Not seen: files read by non-Python subprocesses, and packages outside the repository (.venv)."""
import os
import sys

_LOG = os.environ.get("GENCACHE_LOG")
if _LOG:
    import atexit
    import importlib.abc
    import importlib.util
    import json

    _ROOT = os.environ.get("GENCACHE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    _ROOT = _ROOT.rstrip(os.sep) + os.sep
    _SKIP = (".cache/", ".venv/", ".git/", ".bookkeeping/", ".tools/", "robot/body/bom/gensite/")
    _R, _W, _D = set(), set(), set()
    _QUIET = [0]                     # > 0 inside a wrapped search that records its own key (no folder listings then)

    def _rel(p):
        try:
            s = os.path.abspath(os.fsdecode(p))
        except (TypeError, ValueError):
            return None
        if not s.startswith(_ROOT):
            return None
        r = s[len(_ROOT):]
        if r.startswith(_SKIP) or r.endswith(".lock"):
            return None
        if r.endswith(".pyc"):                              # __pycache__/x.cpython-312.pyc -> x.py
            d, f = os.path.split(r)
            if os.path.basename(d) == "__pycache__":
                r = os.path.join(os.path.dirname(d), f.split(".")[0] + ".py")
        return r

    def note(p, into=_R):
        r = _rel(p)
        if r is not None:
            into.add(r)

    def _audit(event, args):
        try:
            if event == "open":
                mode = args[1] if len(args) > 1 else None
                flags = args[2] if len(args) > 2 else 0
                w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                    (mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
                note(args[0], _W if w else _R)
            elif event in ("os.listdir", "os.scandir", "glob.glob") and _QUIET[0]:
                pass
            elif event in ("os.listdir", "os.scandir"):
                if args and args[0] not in (None, "."):
                    note(args[0], _D)
                elif args:
                    note(os.getcwd(), _D)
            elif event == "glob.glob":
                note(os.path.dirname(os.fsdecode(args[0])) or os.getcwd(), _D)
            elif event == "os.rename":
                note(args[1], _W)
        except Exception:  # noqa: BLE001  an audit hook must never raise
            pass

    sys.addaudithook(_audit)
    if sys.argv and sys.argv[0] not in ("", "-c", "-m"):
        note(sys.argv[0])

    # ---------------------------------------------------------------- native readers (patched right after their module loads)
    def _wrap_fn(mod, name, argi=0):
        f = getattr(mod, name, None)
        if f is None or getattr(f, "_gencache", False):
            return

        def w(*a, **k):
            if len(a) > argi:
                note(a[argi])
            for key in ("file_name", "filename", "path", "file_path"):
                if key in k:
                    note(k[key])
            return f(*a, **k)
        w._gencache = True
        w.__doc__, w.__name__, w.__wrapped__ = f.__doc__, f.__name__, f
        setattr(mod, name, w)

    def _wrap_reader(mod, cls):
        C = getattr(mod, cls, None)
        if C is None or getattr(C, "_gencache", False):
            return
        try:
            class R(C):
                _gencache = True

                def ReadFile(self, p, *a):
                    note(p)
                    return super().ReadFile(p, *a)
            R.__name__ = R.__qualname__ = cls
            setattr(mod, cls, R)
        except TypeError:          # a final class: leave it
            pass

    import re as _re
    _STAMP = _re.compile(rb"(FILE_NAME\s*\(\s*'(?:[^']|'')*'\s*,\s*')\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(')")

    def _stable_step(path):
        try:
            p = os.fsdecode(path)
            with open(p, "rb") as fh:
                b = fh.read()
            head, n = _STAMP.subn(rb"\g<1>2000-01-01T00:00:00\g<2>", b[:4096], count=1)
            if n and head != b[:4096]:
                with open(p, "r+b") as fh:          # same length: the stamp has a fixed width
                    fh.write(head)
        except Exception:  # noqa: BLE001
            pass

    def _wrap_writer(mod, cls):
        C = getattr(mod, cls, None)
        if C is None or getattr(C, "_gencache", False):
            return
        try:
            class Wr(C):
                _gencache = True

                def Write(self, p, *a):
                    r = super().Write(p, *a)
                    _stable_step(p)
                    return r
            Wr.__name__ = Wr.__qualname__ = cls
            setattr(mod, cls, Wr)
        except TypeError:
            pass

    def _wrap_discover(mod):
        f = getattr(mod, "discover", None)
        if f is None or getattr(f, "_gencache", False) or not os.fsdecode(getattr(mod, "__file__", "")).endswith(
                os.path.join("robot", "body", "bom", "definitions.py")):
            return

        def discover(*a, **k):
            root = a[0] if a else k.get("root", getattr(mod, "ROOT", None))
            if root is None or os.path.abspath(os.fsdecode(root)).rstrip(os.sep) + os.sep != _ROOT:
                return f(*a, **k)                          # another tree: plain recording
            _QUIET[0] += 1
            try:
                out = f(*a, **k)
            finally:
                _QUIET[0] -= 1
            for top in getattr(mod, "SEARCH", ["robot"]):
                _D.add(f"glob:{top}/**/*.bom.yaml")        # gencache.dir_state recomputes it: the definition files' paths
            return out
        discover._gencache = True
        discover.__doc__, discover.__name__, discover.__wrapped__ = f.__doc__, f.__name__, f
        mod.discover = discover

    def _patch(name, mod):
        if name.rsplit(".", 1)[-1] == "definitions":
            _wrap_discover(mod)
        elif name == "OCP.STEPControl":
            _wrap_reader(mod, "STEPControl_Reader")
            _wrap_writer(mod, "STEPControl_Writer")
        elif name == "OCP.STEPCAFControl":
            _wrap_reader(mod, "STEPCAFControl_Reader")
            _wrap_writer(mod, "STEPCAFControl_Writer")
        elif name == "OCP.BRepTools":
            B = getattr(mod, "BRepTools", None)
            f = getattr(B, "Read_s", None) if B is not None else None
            if f is not None and not getattr(f, "_gencache", False):
                def read_s(shape, path, *a, _f=f):
                    note(path)
                    return _f(shape, path, *a)
                read_s._gencache = True
                try:
                    B.Read_s = staticmethod(read_s)
                except (TypeError, AttributeError):
                    pass
        elif name == "build123d.importers":
            for n in ("import_step", "import_brep", "import_stl", "import_svg"):
                _wrap_fn(mod, n)
        elif name == "gmsh":
            _wrap_fn(mod, "open"); _wrap_fn(mod, "merge")

    _TARGETS = {"OCP.STEPControl", "OCP.STEPCAFControl", "OCP.BRepTools", "build123d.importers", "gmsh"}

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                _patch(module.__name__, module)
            except Exception:  # noqa: BLE001
                pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS and name.rsplit(".", 1)[-1] != "definitions":
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = _Loader(spec.loader)
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())

    @atexit.register
    def _write():
        try:
            for m in list(sys.modules.values()):
                f = getattr(m, "__file__", None)
                if f:
                    note(f)
            os.makedirs(_LOG, exist_ok=True)
            with open(os.path.join(_LOG, f"{os.getpid()}.json"), "w") as fh:
                json.dump(dict(reads=sorted(_R), writes=sorted(_W), dirs=sorted(_D)), fh)
        except Exception:  # noqa: BLE001
            pass
