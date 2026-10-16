import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 1. Загрузка переменных окружения
load_dotenv()


# 2. Объявление пользовательского инструмента
@tool
def search(query: str) -> str:
    """Инструмент для поиска информации в интернете.

        Args:
            query (str): Поисковый запрос.
    """
    print(f"\n[Инструмент search вызван с аргументом]: '{query}'")
    return f"В Токио сейчас переменная облачность, температура около 18°C, без осадков."


# 3. Инициализация рабочей бесплатной модели через OpenRouter
llm = ChatOllama(
    model="llama3.2",
    # openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    # openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    # max_tokens=1500,
)


# 4. Регистрация инструментов и создание агента
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Запуск агента с кастомным инструментом...")
    query = "Какая погода сейчас в Токио?"
    result = agent.invoke({"messages":[HumanMessage(content=query)]})

    print("---Итоговый ответ агента---")
    print(result["messages"][-1].content)
    print("---------------------------")


if __name__ == "__main__":
    main()
