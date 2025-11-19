"""Telegram бот для RAG системы."""

import logging
import sys
import os
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag_prompt_injection_detector.rag import RAG as RAGClient

from secret import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram бот для работы с RAG."""

    BTN_HELP = "HELP"
    BTN_START = "START"
    BTN_INFO = "INFO"

    def __init__(self, token: str):
        self.token = token
        self.rag = RAGClient()
        self.application = Application.builder().token(token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("info", self.info_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    def get_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Клавиатура с кнопками."""
        keyboard = [
            [KeyboardButton(self.BTN_START), KeyboardButton(self.BTN_HELP), KeyboardButton(self.BTN_INFO)]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    async def start_command(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработка /start."""
        message = (
            "👋 Привет! Я бот для работы с RAG системой.\n\n"
            "💬 Отправь мне вопрос, и я постараюсь на него ответить."
        )
        await update.message.reply_text(message, reply_markup=self.get_main_keyboard())
        logger.info(f"Пользователь {update.effective_user.id} запустил бота")

    async def help_command(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработка /help."""
        message = (
            "📖 Помощь по использованию бота:\n\n"
            "Отправь текстовое сообщение - оно будет передано в RAG систему.\n\n"
            "🔹 START - главная\n"
            "🔹 HELP - помощь\n"
            "🔹 INFO - информация о боте"
        )
        await update.message.reply_text(message, reply_markup=self.get_main_keyboard())

    async def info_command(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработка /info."""
        message = (
            "ℹ️ Информация о боте:\n\n"
            "🤖 Telegram бот для работы с RAG системой обнаружения prompt injection.\n\n"
            "👥 Команда разработчиков:\n"
            "• Strelkov Andrey\n"
            "• Ikonnikov Mark\n"
            "• Prein Dmitry\n"
            "• Korneenko Sergei\n"
            "• Portnov Timyr\n\n"
            "📦 Версия: 0.1.0"
        )
        await update.message.reply_text(message, reply_markup=self.get_main_keyboard())

    async def handle_message(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработка сообщений."""
        user_message = update.message.text
        user_id = update.effective_user.id

        logger.info(f"Сообщение от {user_id}: {user_message}")

        # Обработка кнопок
        if user_message == self.BTN_START:
            await self.start_command(update, context)
            return
        elif user_message == self.BTN_HELP:
            await self.help_command(update, context)
            return
        elif user_message == self.BTN_INFO:
            await self.info_command(update, context)
            return

        # Обработка вопросов
        try:
            response = self.rag.get_response(user_message)
            await update.message.reply_text(
                response, reply_markup=self.get_main_keyboard()
            )
            logger.info(f"Ответ отправлен {user_id}")
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            await update.message.reply_text(
                "⚠️ Ошибка при обработке запроса. Попробуйте позже.",
                reply_markup=self.get_main_keyboard(),
            )

    def run(self) -> None:
        """Запуск бота."""
        logger.info("Запуск бота...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Точка входа."""
    token = TELEGRAM_BOT_TOKEN

    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не установлен!")
        return

    bot = TelegramBot(token)
    bot.run()


if __name__ == "__main__":
    main()