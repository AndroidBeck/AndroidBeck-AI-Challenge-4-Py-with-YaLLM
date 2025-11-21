import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QComboBox,
    QLabel,
)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread

from db import (
    init_db,
    create_conversation,
    get_last_conversation_with_active_messages,
    deactivate_all_messages_for_conversation,
)
from llm_client import MODEL_MAP, get_default_model_name
from chat_logic import chat_turn, summarize_conversation_core


def select_or_create_conversation() -> int:
    last_active = get_last_conversation_with_active_messages()
    if last_active is not None:
        conversation_id, title, created_at, updated_at = last_active
        print(
            f"Continuing conversation #{conversation_id}: '{title}' "
            f"(created_at={created_at}, updated_at={updated_at})"
        )
    else:
        conversation_id = create_conversation()
        print(f"Created new conversation #{conversation_id}")
    return conversation_id


# =========================
# WORKERS (фоновая работа в QThread)
# =========================

class ChatWorker(QObject):
    finished = Signal(object, object, int, object)  # reply_text, usage, messages_sent, error_msg

    def __init__(self, conversation_id: int, user_text: str, model_name: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.user_text = user_text
        self.model_name = model_name

    @Slot()
    def run(self):
        try:
            reply_text, usage, messages_sent = chat_turn(
                self.conversation_id,
                self.user_text,
                self.model_name,
            )
            error_msg = None
        except Exception as e:
            # На всякий случай ловим всё, но основная ошибка уже печатается в chat_turn
            reply_text, usage, messages_sent = None, {}, 0
            error_msg = str(e)

        self.finished.emit(reply_text, usage, messages_sent, error_msg)


class SummaryWorker(QObject):
    finished = Signal(object, object, int, object)  # summary_text, usage, messages_sent, error_msg

    def __init__(self, conversation_id: int, max_tokens: int, model_name: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.max_tokens = max_tokens
        self.model_name = model_name

    @Slot()
    def run(self):
        try:
            summary_text, usage, messages_sent = summarize_conversation_core(
                self.conversation_id,
                self.max_tokens,
                self.model_name,
            )
            error_msg = None
        except Exception as e:
            summary_text, usage, messages_sent = None, {}, 0
            error_msg = str(e)

        self.finished.emit(summary_text, usage, messages_sent, error_msg)


# =========================
# MAIN WINDOW
# =========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Personal AI Project – PySide6 UI")
        self.resize(900, 650)

        # DB + conversation
        init_db()
        self.conversation_id = select_or_create_conversation()

        # current model
        self.current_model_name = get_default_model_name()

        # Хранители текущих потоков, чтобы их не схлопнул GC
        self.current_thread = None
        self.current_worker = None

        # === Widgets ===
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Chat history (read-only)
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        main_layout.addWidget(self.history_view)

        # Top controls: model selector + summary + new conversation
        top_controls = QHBoxLayout()
        main_layout.addLayout(top_controls)

        self.model_combo = QComboBox()
        for idx, name in MODEL_MAP.items():
            self.model_combo.addItem(f"{idx} – {name}", userData=name)
            if name == self.current_model_name:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        top_controls.addWidget(QLabel("Model:"))
        top_controls.addWidget(self.model_combo)

        self.summary_button = QPushButton("Summarize (400 tokens)")
        self.summary_button.clicked.connect(self.on_summarize_clicked)
        top_controls.addWidget(self.summary_button)

        self.new_button = QPushButton("New conversation")
        self.new_button.clicked.connect(self.on_new_conversation_clicked)
        top_controls.addWidget(self.new_button)

        # Bottom area: multiline input + send button
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "Type your message here...\n"
            "(Shift+Enter для новой строки, Enter по кнопке \"Send\")"
        )
        self.input_edit.setFixedHeight(100)
        bottom_layout.addWidget(self.input_edit)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        bottom_layout.addWidget(self.send_button)

        # Status label (tokens, errors)
        self.status_label = QLabel("Ready.")
        main_layout.addWidget(self.status_label)

        # Welcome text
        self.append_system_text("Welcome to Personal AI Project (PySide6 UI).\n")
        self.append_system_text(
            "Commands supported here:\n"
            "  /summarize X, /sum X, /compress X  – summarize active messages\n"
            "  /deactivate                         – deactivate all messages in this conversation\n"
            "You can also use the buttons above."
        )

    # ====== helpers ======

    def append_system_text(self, text: str) -> None:
        self.history_view.append(f"<i>{text}</i>")

    def append_user_text(self, text: str) -> None:
        self.history_view.append(f"<b>YOU:</b> {text}")

    def append_assistant_text(self, text: str) -> None:
        self.history_view.append(f"<b>ASSISTANT:</b>\n{text}")

    def set_busy(self, busy: bool, message: str = "") -> None:
        self.send_button.setEnabled(not busy)
        self.summary_button.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        if busy:
            self.status_label.setText(message or "Working...")
        else:
            if not message:
                message = "Ready."
            self.status_label.setText(message)

    def cleanup_thread(self):
        if self.current_thread is not None:
            self.current_thread.quit()
            self.current_thread.wait()
            self.current_thread = None
        self.current_worker = None

    # ====== slots ======

    def on_model_changed(self, index: int) -> None:
        model_name = self.model_combo.itemData(index)
        if model_name:
            self.current_model_name = model_name
            self.status_label.setText(f"Model changed to: {model_name}")

    def on_new_conversation_clicked(self) -> None:
        self.conversation_id = create_conversation()
        self.append_system_text(f"Started NEW conversation #{self.conversation_id}.")
        self.status_label.setText("New conversation started.")

    def on_summarize_clicked(self) -> None:
        max_tokens = 400
        # Запускаем summary в отдельном потоке
        self.set_busy(True, "Summarizing...")

        thread = QThread()
        worker = SummaryWorker(self.conversation_id, max_tokens, self.current_model_name)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_summary_finished)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.current_thread = thread
        self.current_worker = worker

        thread.start()

    @Slot(object, object, int, object)
    def on_summary_finished(self, summary_text, usage, messages_sent, error_msg):
        self.set_busy(False)

        if messages_sent == 0:
            self.status_label.setText("Nothing to summarize: no active messages.")
            self.cleanup_thread()
            return

        if error_msg is not None:
            self.append_system_text("[ERROR] Summarization failed (see console).")
            self.status_label.setText("Summarization failed (see console).")
            self.cleanup_thread()
            return

        if summary_text is None:
            self.append_system_text("[ERROR] Summarization failed.")
            self.status_label.setText("Summarization failed.")
            self.cleanup_thread()
            return

        self.append_system_text("<b>SUMMARY:</b>")
        self.history_view.append(summary_text)

        self.status_label.setText(
            f"Summary done. input={usage.get('inputTextTokens')}, "
            f"completion={usage.get('completionTokens')}, "
            f"total={usage.get('totalTokens')}, "
            f"messages_sent={messages_sent}"
        )

        self.cleanup_thread()

    def on_send_clicked(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        # Очищаем ввод
        self.input_edit.clear()

        lower = text.lower()

        # Обработка команд (как в CLI)

        if lower == "/deactivate":
            deactivate_all_messages_for_conversation(self.conversation_id)
            self.append_system_text(
                f"All messages in conversation #{self.conversation_id} were deactivated."
            )
            self.status_label.setText("Conversation messages deactivated.")
            return

        if (
            lower.startswith("/summarize")
            or lower.startswith("/compress")
            or lower.startswith("/sum")
        ):
            parts = text.split()
            if len(parts) >= 2:
                try:
                    max_tokens = int(parts[1])
                except ValueError:
                    self.status_label.setText("Invalid token count for summarize.")
                    return
            else:
                max_tokens = 400

            # запуск summarization через кнопку/команду
            self.set_busy(True, "Summarizing...")
            thread = QThread()
            worker = SummaryWorker(self.conversation_id, max_tokens, self.current_model_name)
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self.on_summary_finished)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            self.current_thread = thread
            self.current_worker = worker

            thread.start()
            return

        # Обычное сообщение → показываем сразу в истории
        self.append_user_text(text)

        # Запускаем chat_turn в отдельном потоке
        self.set_busy(True, "Thinking...")

        thread = QThread()
        worker = ChatWorker(self.conversation_id, text, self.current_model_name)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_chat_finished)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.current_thread = thread
        self.current_worker = worker

        thread.start()

    @Slot(object, object, int, object)
    def on_chat_finished(self, reply_text, usage, messages_sent, error_msg):
        self.set_busy(False)

        if error_msg is not None:
            self.append_system_text("[ERROR] LLM call failed (see console).")
            self.status_label.setText("LLM call failed (see console).")
            self.cleanup_thread()
            return

        if reply_text is None:
            self.append_system_text("[ERROR] LLM call failed.")
            self.status_label.setText("LLM call failed.")
            self.cleanup_thread()
            return

        self.append_assistant_text(reply_text)

        self.status_label.setText(
            f"input={usage.get('inputTextTokens')}, "
            f"completion={usage.get('completionTokens')}, "
            f"total={usage.get('totalTokens')}, "
            f"messages_sent={messages_sent}"
        )

        self.cleanup_thread()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
