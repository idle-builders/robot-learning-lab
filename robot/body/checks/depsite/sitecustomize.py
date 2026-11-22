"""Records the files a Python subprocess of a design-check child reads (checks/resultcache.py): on PYTHONPATH only while a
check child runs; writes repository-relative paths to $CHECKS_DEPS_LOG/<pid>.txt at exit. Does nothing otherwise."""
import os
import sys

_LOG = os.environ.get("CHECKS_DEPS_LOG")
if _LOG:
    import atexit

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")) + os.sep
    _SKIP = (".cache/", ".venv/", ".git/", ".bookkeeping/", ".tools/")
    _READ = set()

    def _note(p):
        try:
            s = os.path.abspath(os.fsdecode(p))
        except (TypeError, ValueError):
            return
        if s.startswith(_ROOT) and not s[len(_ROOT):].startswith(_SKIP) and not s.endswith(".pyc"):
            _READ.add(s[len(_ROOT):])

    def _audit(event, args):
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                return
            try:
                _note(args[0])
            except Exception:  # noqa: BLE001
                pass

    sys.addaudithook(_audit)
    if len(sys.argv) and sys.argv[0] not in ("", "-c"):
        _note(sys.argv[0])

    @atexit.register
    def _write():
        try:
            with open(os.path.join(_LOG, f"{os.getpid()}.txt"), "w") as fh:
                fh.write("\n".join(sorted(_READ)))
        except OSError:
            pass
