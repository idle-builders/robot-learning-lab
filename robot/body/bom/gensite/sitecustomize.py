"""Records what a generator process reads and writes, for the generator cache (robot/body/bom/gencache.py, make.py).

On PYTHONPATH only while make.py runs a generator ($GENCACHE_LOG set); does nothing otherwise. Every Python process of the
run (the generator and its Python subprocesses) writes $GENCACHE_LOG/<pid>.json at exit with repository-relative paths:
- reads: files opened for reading (Python's `open` audit event: json, numpy, trimesh, every module source), plus the files
  handed to native readers whose reads Python never sees: build123d import_step / import_brep / import_stl, OpenCascade's
  STEPControl_Reader / STEPCAFControl_Reader.ReadFile and BRepTools.Read_s, gmsh.open / gmsh.merge;
- modules: the source file of every module under the repository loaded at exit (a module loaded from __pycache__ never opens
  its .py, so the source is recorded from sys.modules);
- dirs: folders listed (os.listdir / os.scandir / glob), whose contents can change a result;
- writes: files opened for writing or renamed into place (used to name undeclared outputs).
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

    def _patch(name, mod):
        if name == "OCP.STEPControl":
            _wrap_reader(mod, "STEPControl_Reader")
        elif name == "OCP.STEPCAFControl":
            _wrap_reader(mod, "STEPCAFControl_Reader")
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
            if name not in _TARGETS:
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
