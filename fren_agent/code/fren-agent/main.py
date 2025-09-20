import os
import sys
from PyQt5 import QtWidgets

def _preflight_import_or_die(project_root: str):
    # Make 'src' importable and attempt to import agent module before GUI starts.
    sys.path.insert(0, os.path.join(project_root, "src"))
    try:
        import importlib
        mod = importlib.import_module("fren_agent.agent")
        # sanity check symbols
        if not hasattr(mod, "AgentConfig") or not hasattr(mod, "FrenAgent"):
            raise ImportError("fren_agent.agent imported but missing AgentConfig/FrenAgent (module likely half-initialized).")
    except Exception as e:
        import traceback
        print("\n[preflight] Import of fren_agent.agent failed. Full traceback:\n", file=sys.stderr)
        traceback.print_exc()
        print("\n[preflight] Fix the above error, then re-run. Exiting.", file=sys.stderr)
        sys.exit(1)

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    _preflight_import_or_die(root)  # show real cause in console if import breaks

    # Import GUI only after agent preflight passes
    from fren_agent.gui.app import FrenGUI

    cfg_path = os.path.join(root, "config", "config.yaml")
    os.makedirs(os.path.join(root, "assets"), exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    gui = FrenGUI(cfg_path)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
