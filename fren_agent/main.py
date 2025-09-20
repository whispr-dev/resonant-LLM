import os
import sys
from PyQt5 import QtWidgets
from src.fren_agent.gui.app import FrenGUI

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(root, "config", "config.yaml")
    os.makedirs(os.path.join(root, "assets"), exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    gui = FrenGUI(cfg_path)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
