__version__ = "0.1.0"

from rag_prompt_injection_detector.rag import RAG
def check_rag_working():
    """
    Функция для проверки работоспособности RAG-системы.
    Создаёт экземпляр и выполняет тестовый запрос.
    """
    try:
        # Создаём экземпляр RAG
        rag = RAG()

        # Выполняем тестовый запрос
        test_query = "How do I generate random numbers in Python?"
        response = rag.get_response(test_query)

        print("✓ RAG успешно инициализирован!")
        print(f"Запрос: {test_query}")
        print(f"Ответ: {response}")

        return True

    except Exception as e:
        print("✗ Ошибка при проверке RAG:")
        print(f"{type(e).__name__}: {e}")
        return False


# Проверка при импорте модуля
if __name__ == "__main__":
    check_rag_working()
