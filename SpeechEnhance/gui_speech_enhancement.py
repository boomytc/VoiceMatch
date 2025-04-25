# -*- coding: utf-8 -*-
"""
基于 PyQt6 的图形界面
1. 选择音频文件
2. 选择执行语音增强 / 超分辨率
3. 调用 speech_enhancement.process_file 处理
4. 显示处理前 / 处理后的波形与频谱
"""

import sys
import os
import threading
from typing import Tuple

import numpy as np
import librosa
import librosa.display  # type: ignore
import matplotlib
# 使用非交互式后端，随后嵌入到 Qt
matplotlib.use("Agg")
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # matplotlib>=3.5
except ImportError:  # 兼容旧版本
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QCheckBox,
    QMessageBox,
    QSizePolicy,
    QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# 导入处理函数
sys.path.append(os.path.dirname(__file__))
from speech_enhancement import process_file  # noqa: E402


class Worker(QObject):
    """后台线程执行音频处理"""

    finished = pyqtSignal(bool, str, str)  # success, output_path, error_msg

    def __init__(self, args: Tuple[str, int, bool, bool, str | None]):
        super().__init__()
        self.args = args

    def run(self):
        try:
            success, out_path = process_file(self.args)
            self.finished.emit(success, out_path if out_path else "", "")
        except Exception as e:  # pragma: no cover
            self.finished.emit(False, "", str(e))


class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 4, height: float = 3, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def clear(self):
        self.ax.clear()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Enhancement GUI")
        self.file_path: str | None = None
        self.out_path: str | None = None

        self._init_ui()

    # ----------------------------- UI -----------------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # === 顶部控件 ===
        ctrl_layout = QHBoxLayout()
        self.btn_select = QPushButton("选择音频文件…")
        self.btn_select.clicked.connect(self.select_file)
        self.lbl_path = QLabel("未选择文件")
        self.chk_se = QCheckBox("语音增强")
        self.chk_se.setChecked(True)
        self.chk_sr = QCheckBox("超分辨率")
        self.chk_sr.setChecked(False)
        self.btn_run = QPushButton("开始处理")
        self.btn_run.clicked.connect(self.start_processing)

        ctrl_layout.addWidget(self.btn_select)
        ctrl_layout.addWidget(self.lbl_path, stretch=1)
        ctrl_layout.addWidget(self.chk_se)
        ctrl_layout.addWidget(self.chk_sr)
        ctrl_layout.addWidget(self.btn_run)
        main_layout.addLayout(ctrl_layout)

        # === 状态标签 ===
        self.lbl_status = QLabel("")
        main_layout.addWidget(self.lbl_status)

        # === 绘图区域 ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # 原始
        orig_widget = QWidget()
        orig_layout = QVBoxLayout(orig_widget)
        self.orig_wave = MplCanvas()
        self.orig_spec = MplCanvas()
        orig_layout.addWidget(self.orig_wave)
        orig_layout.addWidget(self.orig_spec)
        splitter.addWidget(orig_widget)

        # 处理后
        proc_widget = QWidget()
        proc_layout = QVBoxLayout(proc_widget)
        self.proc_wave = MplCanvas()
        self.proc_spec = MplCanvas()
        proc_layout.addWidget(self.proc_wave)
        proc_layout.addWidget(self.proc_spec)
        splitter.addWidget(proc_widget)
        # 在添加完所有 widget 后设置初始大小为均分
        # 获取一个较大的初始宽度，避免窗口过小导致计算不准
        initial_width = self.sizeHint().width() if self.sizeHint().width() > 800 else 1200
        splitter.setSizes([initial_width // 2, initial_width // 2])

    # --------------------------- 业务逻辑 ---------------------------
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg)")
        if path:
            self.file_path = path
            self.lbl_path.setText(os.path.basename(path))
            self.lbl_status.setText("")
            # 清空图形
            for c in (self.orig_wave, self.orig_spec, self.proc_wave, self.proc_spec):
                c.clear()

    def start_processing(self):
        if not self.file_path:
            QMessageBox.warning(self, "提示", "请先选择音频文件！")
            return
        do_se = self.chk_se.isChecked()
        do_sr = self.chk_sr.isChecked()
        if not do_se and not do_sr:
            QMessageBox.warning(self, "提示", "请至少选择一种任务（语音增强或超分辨率）！")
            return

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("处理中，请稍候…")
        QApplication.processEvents()

        # GPU id 固定 0，输出路径 None
        args = (self.file_path, 0, do_se, do_sr, None)
        # 使用线程避免阻塞界面
        worker = Worker(args)
        thread = threading.Thread(target=worker.run, daemon=True)
        worker.finished.connect(self.on_finished)  # type: ignore[arg-type]
        thread.start()

    def on_finished(self, success: bool, out_path: str, err: str):  # noqa: D401
        self.btn_run.setEnabled(True)
        if not success:
            self.lbl_status.setText(f"处理失败: {err}")
            QMessageBox.critical(self, "错误", f"处理失败: {err}")
            return

        self.out_path = out_path
        self.lbl_status.setText(f"处理完成: {out_path}")
        # 绘制前后波形+频谱
        try:
            self._draw_plots()
        except Exception as e:  # pragma: no cover
            QMessageBox.warning(self, "绘图错误", str(e))

    # --------------------------- 绘图 ---------------------------
    def _draw_plots(self):
        if not self.file_path or not self.out_path:
            return
        y_orig, sr_orig = librosa.load(self.file_path, sr=None, mono=True)
        y_proc, sr_proc = librosa.load(self.out_path, sr=None, mono=True)

        # 波形
        self._plot_wave(self.orig_wave, y_orig, sr_orig, "原始波形")
        self._plot_wave(self.proc_wave, y_proc, sr_proc, "处理后波形")
        # 频谱
        self._plot_spec(self.orig_spec, y_orig, sr_orig, "原始频谱")
        self._plot_spec(self.proc_spec, y_proc, sr_proc, "处理后频谱")

    @staticmethod
    def _plot_wave(canvas: MplCanvas, y: np.ndarray, sr: int, title: str):
        canvas.ax.clear()
        t = np.linspace(0, len(y) / sr, num=len(y))
        canvas.ax.plot(t, y, color="steelblue")
        canvas.ax.set_title(title)
        canvas.ax.set_xlabel("Time [s]")
        canvas.ax.set_ylabel("Amplitude")
        canvas.fig.tight_layout()
        canvas.draw()

    @staticmethod
    def _plot_spec(canvas: MplCanvas, y: np.ndarray, sr: int, title: str):
        canvas.ax.clear()
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="linear",
            cmap="magma",
            ax=canvas.ax,
        )
        canvas.ax.set_title(title)
        canvas.fig.colorbar(img, ax=canvas.ax, format="%+2.0f dB")
        canvas.fig.tight_layout()
        canvas.draw()


# ------------------------------- 入口 -------------------------------

def main():  # pragma: no cover
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()