import os
import cv2
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

from ..agent import load_config, FrenAgent

class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, str)  # frame (numpy), caption

    def __init__(self, agent: FrenAgent, fps: int):
        super().__init__()
        self.agent = agent
        self.interval_ms = max(1, int(1000 / max(1, fps)))
        self._running = True

    def run(self):
        while self._running:
            caption = self.agent.read_frame_and_caption()
            ok, frame = self.agent.cap.read()
            if ok and frame is not None:
                self.frame_ready.emit(frame, caption)
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
        self.wait(500)

class FrenGUI(QtWidgets.QMainWindow):
    def __init__(self, cfg_path: str):
        super().__init__()
        self.setWindowTitle("Fren-Agent")
        self.resize(1100, 720)
        self.cfg = load_config(cfg_path)
        self.agent = FrenAgent(self.cfg)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Tab: Agent
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #111;")

        self.caption_label = QtWidgets.QLabel("Caption: ...")
        self.caption_label.setWordWrap(True)

        self.input_edit = QtWidgets.QTextEdit()
        self.input_edit.setPlaceholderText("Type to talk with Fren-Agent...")

        self.talk_btn = QtWidgets.QPushButton("Speak")
        self.talk_btn.clicked.connect(self.on_speak_clicked)

        self.listen_btn = QtWidgets.QPushButton("Listen (PTT 4s)")
        self.listen_btn.clicked.connect(self.on_listen_clicked)

        self.reply_view = QtWidgets.QTextBrowser()

        agent_layout = QtWidgets.QGridLayout()
        agent_layout.addWidget(self.video_label, 0, 0, 1, 2)
        agent_layout.addWidget(self.caption_label, 1, 0, 1, 2)
        agent_layout.addWidget(self.input_edit, 2, 0, 1, 2)
        agent_layout.addWidget(self.talk_btn, 3, 0, 1, 1)
        agent_layout.addWidget(self.listen_btn, 3, 1, 1, 1)
        agent_layout.addWidget(self.reply_view, 4, 0, 1, 2)

        agent_tab = QtWidgets.QWidget()
        agent_tab.setLayout(agent_layout)
        tabs.addTab(agent_tab, "Agent")

        # Tab: Settings
        self.cfg_text = QtWidgets.QPlainTextEdit()
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg_text.setPlainText(f.read())
        self.save_cfg_btn = QtWidgets.QPushButton("Save Config")
        self.save_cfg_btn.clicked.connect(lambda: self.on_save_config(cfg_path))

        set_layout = QtWidgets.QVBoxLayout()
        set_layout.addWidget(self.cfg_text)
        set_layout.addWidget(self.save_cfg_btn)
        set_tab = QtWidgets.QWidget()
        set_tab.setLayout(set_layout)
        tabs.addTab(set_tab, "Settings")

        # Tab: Logs (minimal)
        self.log_view = QtWidgets.QTextBrowser()
        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(self.log_view)
        log_tab.setLayout(log_layout)
        tabs.addTab(log_tab, "Logs")

        # threads
        self.vthread = VideoThread(self.agent, self.cfg.fps)
        self.vthread.frame_ready.connect(self.on_frame)
        self.vthread.start()

    def on_frame(self, frame, caption: str):
        # draw frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(self.video_label.width(),
                                                  self.video_label.height(),
                                                  QtCore.Qt.KeepAspectRatio,
                                                  QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        if caption:
            self.caption_label.setText(f"Caption: {caption}")

    def on_speak_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self.log("[user] " + text)
        reply = self.agent.respond(text)
        self.reply_view.append(reply)
        self.agent.say(reply)
        self.input_edit.clear()

    def on_listen_clicked(self):
        self.log("[info] recording 4 seconds (PTT)...")
        text = self.agent.stt_once(4.0, "./assets/utt.wav")
        self.log("[stt] " + text)
        if text:
            reply = self.agent.respond(text)
            self.reply_view.append(reply)
            self.agent.say(reply)

    def on_save_config(self, cfg_path: str):
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(self.cfg_text.toPlainText())
        self.log("[cfg] saved. restart recommended for model changes.")

    def log(self, s: str):
        self.log_view.append(s)

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.vthread.stop()
        except Exception:
            pass
        try:
            self.agent.release()
        except Exception:
            pass
        e.accept()
