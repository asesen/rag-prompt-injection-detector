# Telegram Bot для RAG

Простой бот для взаимодействия с RAG системой.

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

1. Получите токен у [@BotFather](https://t.me/BotFather)
2. Установите токен:
```bash
export TELEGRAM_BOT_TOKEN='your_token'
```
3. Запустите:
```bash
./run_bot.sh
```

## Интеграция с RAG

Откройте `rag_client.py` и замените метод `get_response()` на вашу логику:

```python
def get_response(self, query: str) -> str:
    # Ваша RAG система
    result = your_rag_system.process(query)
    return result
```

## Структура

- `bot.py` - основной код бота
- `rag_client.py` - интерфейс для RAG системы
