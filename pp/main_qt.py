import sys
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QComboBox,
    QLabel,
)
from PySide6.QtCore import Qt

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Personal AI Project – PySide6 UI")
        self.resize(800, 600)

        # DB + conversation
        init_db()
        self.conversation_id = select_or_create_conversation()

        # current model
        self.current_model_name = get_default_model_name()

        # === Widgets ===
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Chat history
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        main_layout.addWidget(self.history_view)

        # Model + summary line
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

        # Кнопка summarize
        self.summary_button = QPushButton("Summarize (400 tokens)")
        self.summary_button.clicked.connect(self.on_summarize_clicked)
        top_controls.addWidget(self.summary_button)

        # Кнопка new conversation
        self.new_button = QPushButton("New conversation")
        self.new_button.clicked.connect(self.on_new_conversation_clicked)
        top_controls.addWidget(self.new_button)

        # Полоса ввода
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Type your message...")
        self.input_edit.returnPressed.connect(self.on_send_clicked)
        bottom_layout.addWidget(self.input_edit)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        bottom_layout.addWidget(self.send_button)

        # Status label (tokens, errors)
        self.status_label = QLabel("Ready.")
        main_layout.addWidget(self.status_label)

        # Приветственное сообщение
        self.append_system_text("Welcome to Personal AI Project (PySide6 UI).\n")

    # ====== helpers ======

    def append_system_text(self, text: str) -> None:
        self.history_view.append(f"<i>{text}</i>")

    def append_user_text(self, text: str) -> None:
        self.history_view.append(f"<b>YOU:</b> {text}")

    def append_assistant_text(self, text: str) -> None:
        # Простой вариант, без markdown
        self.history_view.append(f"<b>ASSISTANT:</b>\n{text}")

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
        summary_text, usage, messages_sent = summarize_conversation_core(
            self.conversation_id,
            max_tokens,
            self.current_model_name,
        )
        if summary_text is None:
            self.status_label.setText("Summarization failed.")
            return

        self.append_system_text("<b>SUMMARY:</b>")
        self.history_view.append(summary_text)

        self.status_label.setText(
            f"Summary done. input={usage.get('inputTextTokens')}, "
            f"completion={usage.get('completionTokens')}, "
            f"total={usage.get('totalTokens')}, "
            f"messages_sent={messages_sent}"
        )

    def on_send_clicked(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return

        self.input_edit.clear()

        # Поддержим некоторые команды, как в CLI
        lower = text.lower()

        if lower == "/deactivate":
            deactivate_all_messages_for_conversation(self.conversation_id)
            self.append_system_text(
                f"All messages in conversation #{self.conversation_id} were deactivated."
            )
            self.status_label.setText("Conversation messages deactivated.")
            return

        if lower.startswith("/summarize") or lower.startswith("/compress") or lower.startswith("/sum"):
            parts = text.split()
            if len(parts) >= 2:
                try:
                    max_tokens = int(parts[1])
                except ValueError:
                    self.status_label.setText("Invalid token count for summarize.")
                    return
            else:
                max_tokens = 400

            summary_text, usage, messages_sent = summarize_conversation_core(
                self.conversation_id,
                max_tokens,
                self.current_model_name,
            )

            if summary_text is None:
                self.status_label.setText("Summarization failed.")
                return

            self.append_system_text("<b>SUMMARY:</b>")
            self.history_view.append(summary_text)

            self.status_label.setText(
                f"Summary done. input={usage.get('inputTextTokens')}, "
                f"completion={usage.get('completionTokens')}, "
                f"total={usage.get('totalTokens')}, "
                f"messages_sent={messages_sent}"
            )
            return

        # Обычное сообщение
        self.append_user_text(text)

        # Важно: сейчас вызываем LLM синхронно → UI подфризит.
        reply_text, usage, messages_sent = chat_turn(
            self.conversation_id,
            text,
            self.current_model_name,
        )

        if reply_text is None:
            self.append_system_text("[ERROR] LLM call failed.")
            self.status_label.setText("Error calling LLM.")
            return

        self.append_assistant_text(reply_text)

        self.status_label.setText(
            f"input={usage.get('inputTextTokens')}, "
            f"completion={usage.get('completionTokens')}, "
            f"total={usage.get('totalTokens')}, "
            f"messages_sent={messages_sent}"
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
