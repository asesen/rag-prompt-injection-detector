В папке rag_prompt_injection_detector находятся основные модули нашего RAG агента:

-rag класс RAG - основа агента, в нем логика работы, llm, emb

-detector - Detector класс, который используется для детекции jailbreak

-vectorstore - VectorStore класс, который используется для получения context, jailbreak context, a также с его помощью можно обучить Detector

-primary_cpu.pt - хранит обученный детектор

В папке data храняться данные, полученные нами после обработки исходного датасета
