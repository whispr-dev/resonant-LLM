import sys, traceback, importlib, os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)  # project root
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

print("[debug] sys.path[0]:", sys.path[0])
print("[debug] trying: import fren_agent.agent ...")
try:
    mod = importlib.import_module("fren_agent.agent")
    print("[debug] import OK.")
    has = {name: hasattr(mod, name) for name in ("AgentConfig", "FrenAgent")}
    print("[debug] symbols:", json.dumps(has))
    print("[debug] done.")
except Exception:
    print("[debug] import FAILED with traceback:\n")
    traceback.print_exc()
    sys.exit(1)
