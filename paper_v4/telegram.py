"""Minimal Telegram outbox delivery; secrets are read only from environment."""

from __future__ import annotations

import os

import requests

from paper_v4.store import PaperStore


PREFIX = "🧪 CRYPTO V7 EXPERIMENTAL PAPER\nNO REAL ORDERS\n"


class TelegramNotifier:
    def __init__(self, store: PaperStore):
        self.store = store
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def flush(self) -> None:
        if not self.configured:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for row in self.store.pending_notifications():
            try:
                response = requests.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": PREFIX + row["message"],
                        "disable_web_page_preview": True,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                if not payload.get("ok"):
                    raise RuntimeError(str(payload.get("description", "Telegram rejected message")))
                self.store.notification_sent(row["id"])
            except Exception as error:
                self.store.notification_failed(row["id"], str(error))
                break
