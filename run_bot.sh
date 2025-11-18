#!/bin/bash
# Скрипт для запуска Telegram бота

# Проверяем наличие токена
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ Ошибка: Не установлена переменная окружения TELEGRAM_BOT_TOKEN"
    echo ""
    echo "Установите токен командой:"
    echo "export TELEGRAM_BOT_TOKEN='your_token_here'"
    echo ""
    echo "Получить токен можно у @BotFather в Telegram"
    exit 1
fi

# Активируем виртуальное окружение
if [ -d "venv" ]; then
    echo "📦 Активация виртуального окружения..."
    source venv/bin/activate
fi

echo "🚀 Запуск Telegram бота..."
python -m telegram_bot.bot

