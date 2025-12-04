"""
PySide6 GUI for the RAG agent.

Features (baseline):
- Tabs: Upload / Query / Settings
- PDF-only upload with page range, metadata, replace flag
- Query with product/model multi-select, top_k & rerank sliders, reranker toggle
- Settings: .env path selection & reload, Qdrant status indicator

Note: PDF rendering with bbox highlight is stubbed as a simple text viewer; extend with Qt PDF modules if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from PySide6 import QtCore, QtWidgets

from rag_agent import RAGAgent, RAGConfig


# ----------------- Worker threads -----------------
class IngestWorker(QtCore.QThread):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        agent: RAGAgent,
        files: List[Path],
        session: str,
        product_name: Optional[str],
        model_id: Optional[str],
        version: Optional[str],
        region: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
        replace: bool,
    ):
        super().__init__()
        self.agent = agent
        self.files = files
        self.session = session
        self.product_name = product_name
        self.model_id = model_id
        self.version = version
        self.region = region
        self.page_start = page_start
        self.page_end = page_end
        self.replace = replace

    def run(self):
        try:
            self.agent.ingest_files(
                files=self.files,
                session_id=self.session,
                product_name=self.product_name,
                model_id=self.model_id,
                version=self.version,
                region=self.region,
                page_start=self.page_start,
                page_end=self.page_end,
                replace_existing=self.replace,
            )
            self.finished.emit(f"Ingested {len(self.files)} file(s) into session '{self.session}'.")
        except Exception as e:
            self.failed.emit(str(e))


class QueryWorker(QtCore.QThread):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        agent: RAGAgent,
        session: str,
        question: str,
        product_names: List[str],
        model_ids: List[str],
        top_k: int,
    ):
        super().__init__()
        self.agent = agent
        self.session = session
        self.question = question
        self.product_names = product_names or None
        self.model_ids = model_ids or None
        self.top_k = top_k

    def run(self):
        try:
            res = self.agent.query(
                question=self.question,
                session_id=self.session,
                product_names=self.product_names,
                model_ids=self.model_ids,
                top_k=self.top_k,
            )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


# ----------------- GUI -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Datasheet QA")
        self.resize(1100, 800)

        self.agent: Optional[RAGAgent] = None
        self.env_path = Path(".env")

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_upload_tab()
        self._init_query_tab()
        self._init_settings_tab()
        self._workers: List[QtCore.QThread] = []

        self._load_env(self.env_path)
        self._init_agent()
        self._apply_button_style()

    # ---- Tabs ----
    def _init_upload_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        self.session_input = QtWidgets.QLineEdit("demo")
        self.file_button = QtWidgets.QPushButton("Select PDF(s)")
        self.file_button.clicked.connect(self._choose_files)
        self.file_label = QtWidgets.QLabel("No files selected")

        self.page_start_input = QtWidgets.QSpinBox()
        self.page_start_input.setMinimum(1)
        self.page_start_input.setMaximum(10000)
        self.page_end_input = QtWidgets.QSpinBox()
        self.page_end_input.setMinimum(1)
        self.page_end_input.setMaximum(10000)

        self.replace_check = QtWidgets.QCheckBox("Replace existing doc_id")
        self.product_input = QtWidgets.QLineEdit()
        self.model_input = QtWidgets.QLineEdit()
        self.version_input = QtWidgets.QLineEdit()
        self.region_input = QtWidgets.QLineEdit()

        self.upload_button = QtWidgets.QPushButton("Ingest / Embed")
        self.upload_button.clicked.connect(self._run_ingest)

        layout.addRow("Session ID", self.session_input)
        layout.addRow("PDF files", self.file_button)
        layout.addRow("", self.file_label)
        layout.addRow("Page start", self.page_start_input)
        layout.addRow("Page end", self.page_end_input)
        layout.addRow(self.replace_check)
        layout.addRow("Product name", self.product_input)
        layout.addRow("Model ID", self.model_input)
        layout.addRow("Version", self.version_input)
        layout.addRow("Region", self.region_input)
        layout.addRow(self.upload_button)

        layout.setVerticalSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)

        w.setLayout(layout)
        self.tabs.addTab(w, "업로드")

        self.selected_files: List[Path] = []

    def _init_query_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        self.query_session_input = QtWidgets.QLineEdit("demo")
        self.question_input = QtWidgets.QTextEdit()

        # Product/Model multi-select via list widgets
        self.prod_entry = QtWidgets.QLineEdit()
        self.prod_add_btn = QtWidgets.QPushButton("Add product")
        self.prod_add_btn.setMinimumHeight(32)
        self.prod_add_btn.setStyleSheet("font-size: 10pt;")
        self.prod_list = QtWidgets.QListWidget()
        self.prod_add_btn.clicked.connect(self._add_product)
        prod_box = QtWidgets.QHBoxLayout()
        prod_box.addWidget(self.prod_entry)
        prod_box.addWidget(self.prod_add_btn)

        self.model_entry = QtWidgets.QLineEdit()
        self.model_add_btn = QtWidgets.QPushButton("Add model")
        self.model_add_btn.setMinimumHeight(32)
        self.model_add_btn.setStyleSheet("font-size: 10pt;")
        self.model_list = QtWidgets.QListWidget()
        self.model_add_btn.clicked.connect(self._add_model)
        model_box = QtWidgets.QHBoxLayout()
        model_box.addWidget(self.model_entry)
        model_box.addWidget(self.model_add_btn)

        self.topk_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.topk_slider.setMinimum(4)
        self.topk_slider.setMaximum(32)
        self.topk_slider.setValue(12)
        self.topk_label = QtWidgets.QLabel("top_k: 12")
        self.topk_slider.valueChanged.connect(lambda v: self.topk_label.setText(f"top_k: {v}"))

        self.rerank_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rerank_slider.setMinimum(2)
        self.rerank_slider.setMaximum(10)
        self.rerank_slider.setValue(4)
        self.rerank_label = QtWidgets.QLabel("rerank_top_n: 4")
        self.rerank_slider.valueChanged.connect(lambda v: self.rerank_label.setText(f"rerank_top_n: {v}"))

        self.rerank_toggle = QtWidgets.QCheckBox("Use reranker (default On)")
        self.rerank_toggle.setChecked(True)

        self.query_button = QtWidgets.QPushButton("질의 실행")
        self.query_button.clicked.connect(self._run_query)
        self.clear_answer_button = QtWidgets.QPushButton("답변 내용 지우기")
        self.clear_answer_button.clicked.connect(self._clear_answer)
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addWidget(self.query_button)
        btn_box.addWidget(self.clear_answer_button)

        self.answer_view = QtWidgets.QTextEdit()
        self.answer_view.setReadOnly(True)

        layout.addRow("Session ID", self.query_session_input)
        layout.addRow("Question", self.question_input)
        layout.addRow("Product names", prod_box)
        self.prod_list.setMaximumHeight(60)
        layout.addRow(self.prod_list)
        layout.addRow("Model IDs", model_box)
        self.model_list.setMaximumHeight(60)
        layout.addRow(self.model_list)
        layout.addRow(self.topk_label, self.topk_slider)
        layout.addRow(self.rerank_label, self.rerank_slider)
        layout.addRow(self.rerank_toggle)
        layout.addRow(btn_box)
        self.answer_view.setMinimumHeight(600)
        layout.addRow("Answer / Sources", self.answer_view)

        w.setLayout(layout)
        self.tabs.addTab(w, "질의")

    def _init_settings_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()

        self.env_path_label = QtWidgets.QLabel(str(self.env_path.resolve()))
        self.env_button = QtWidgets.QPushButton(".env 선택")
        self.env_button.clicked.connect(self._choose_env)
        env_box = QtWidgets.QHBoxLayout()
        env_box.addWidget(self.env_path_label)
        env_box.addWidget(self.env_button)

        self.env_reload_btn = QtWidgets.QPushButton(".env 재로딩")
        self.env_reload_btn.clicked.connect(lambda: self._load_env(self.env_path))

        self.qdrant_status = QtWidgets.QLabel("Qdrant: unknown")

        layout.addRow(".env 경로", env_box)
        layout.addRow(self.env_reload_btn)
        layout.addRow("Qdrant 상태", self.qdrant_status)

        w.setLayout(layout)
        self.tabs.addTab(w, "설정")

    def _apply_button_style(self):
        btns = []
        for name in [
            "file_button",
            "upload_button",
            "query_button",
            "clear_answer_button",
            "env_button",
            "env_reload_btn",
        ]:
            b = getattr(self, name, None)
            if b:
                btns.append(b)
        for b in btns:
            b.setMinimumHeight(36)
            f = b.font()
            f.setPointSize(14)
            b.setFont(f)

    # ---- Actions ----
    def _choose_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select PDF files", "", "PDF Files (*.pdf)")
        if paths:
            self.selected_files = [Path(p) for p in paths]
            self.file_label.setText("\n".join(paths))

    def _choose_env(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .env", "", "Env Files (*.env);;All Files (*)")
        if path:
            self.env_path = Path(path)
            self.env_path_label.setText(str(self.env_path.resolve()))
            self._load_env(self.env_path)
            self._init_agent()

    def _load_env(self, path: Path):
        if path.exists():
            load_dotenv(path)
            self._log(f".env loaded from {path}")
        else:
            self._log(f".env not found at {path}", error=True)

    def _init_agent(self):
        # Close previous embedded client to avoid qdrant_local lock when reloading .env
        try:
            if self.agent and getattr(self.agent, "qdrant", None):
                self.agent.qdrant.close()
        except Exception:
            pass
        try:
            self.agent = RAGAgent(config=RAGConfig())
            # Reflect actual qdrant mode reported by the agent
            mode = getattr(self.agent, "qdrant_mode", "http")
            self.qdrant_status.setText(f"Qdrant: {mode}")
        except Exception as e:
            self.agent = None
            self.qdrant_status.setText("Qdrant: unavailable")
            self._log(f"Failed to init agent: {e}", error=True)

    def _run_ingest(self):
        if not self.agent:
            self._log("Agent not initialized", error=True)
            return
        if not self.selected_files:
            QtWidgets.QMessageBox.warning(self, "Missing files", "Select at least one PDF.")
            return
        session = self.session_input.text().strip()
        if not session:
            QtWidgets.QMessageBox.warning(self, "Missing session", "Session ID is required.")
            return
        ps = self.page_start_input.value()
        pe = self.page_end_input.value()
        if pe < ps:
            QtWidgets.QMessageBox.warning(self, "Invalid pages", "End page must be >= start page.")
            return
        worker = IngestWorker(
            agent=self.agent,
            files=self.selected_files,
            session=session,
            product_name=self.product_input.text().strip() or None,
            model_id=self.model_input.text().strip() or None,
            version=self.version_input.text().strip() or None,
            region=self.region_input.text().strip() or None,
            page_start=ps,
            page_end=pe,
            replace=self.replace_check.isChecked(),
        )
        self._workers.append(worker)
        worker.finished.connect(lambda msg, w=worker: (self._log(msg), self._cleanup_worker(w)))
        worker.failed.connect(lambda err, w=worker: (self._log(f"Ingest failed: {err}", error=True), self._cleanup_worker(w)))
        worker.start()

    def _run_query(self):
        if not self.agent:
            self._log("Agent not initialized", error=True)
            return
        session = self.query_session_input.text().strip()
        if not session:
            QtWidgets.QMessageBox.warning(self, "Missing session", "Session ID is required.")
            return
        question = self.question_input.toPlainText().strip()
        if not question:
            QtWidgets.QMessageBox.warning(self, "Missing question", "Enter a question.")
            return
        products = [self.prod_list.item(i).text() for i in range(self.prod_list.count())]
        models = [self.model_list.item(i).text() for i in range(self.model_list.count())]
        top_k = self.topk_slider.value()
        # rerank toggle affects cfg use_reranker
        self.agent.cfg.use_reranker = self.rerank_toggle.isChecked()
        self.agent.cfg.rerank_top_n = self.rerank_slider.value()

        worker = QueryWorker(
            agent=self.agent,
            session=session,
            question=question,
            product_names=products,
            model_ids=models,
            top_k=top_k,
        )
        self._workers.append(worker)
        worker.finished.connect(lambda res, w=worker: (self._show_answer(res), self._cleanup_worker(w)))
        worker.failed.connect(lambda err, w=worker: (self._log(f"Query failed: {err}", error=True), self._cleanup_worker(w)))
        worker.start()

    def _add_product(self):
        txt = self.prod_entry.text().strip()
        if txt:
            self.prod_list.addItem(txt)
            self.prod_entry.clear()

    def _add_model(self):
        txt = self.model_entry.text().strip()
        if txt:
            self.model_list.addItem(txt)
            self.model_entry.clear()

    def _show_answer(self, res: dict):
        ans = res.get("answer", "")
        contexts = res.get("contexts", [])
        src_lines = []
        for i, ctx in enumerate(contexts, 1):
            payload = ctx.get("payload", {})
            doc = payload.get("doc_id", "unknown")
            page = payload.get("page", "?")
            score = ctx.get("score", 0)
            src_lines.append(f"[{i}] {doc} p{page} (score {score:.3f})")
        block = ans + "\n\nSources:\n" + "\n".join(src_lines)
        # Append to preserve previous answers; separate with a line.
        if self.answer_view.toPlainText().strip():
            self.answer_view.append("\n---\n")
        self.answer_view.append(block)
        self._log("Query completed.")

    def _clear_answer(self):
        self.answer_view.clear()
        self._log("Cleared answers.")

    def _log(self, msg: str, error: bool = False):
        prefix = "[ERR] " if error else "[INFO] "
        print(prefix + msg)

    def _cleanup_worker(self, worker: QtCore.QThread):
        try:
            if worker.isRunning():
                worker.quit()
                worker.wait(2000)
        except Exception:
            pass
        try:
            worker.deleteLater()
        except Exception:
            pass
        if worker in self._workers:
            self._workers.remove(worker)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
