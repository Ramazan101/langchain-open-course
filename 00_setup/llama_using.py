from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.0
)

messages = [
    SystemMessage(content="Ты строгий предприниматель в отделе видеокарт."),
    HumanMessage(content="Я хочу купить 4 видеокарты для GTA6! Посчитай сумму ИИ быстро."),
    AIMessage(content="Ваши карты будут стоит 800$ за все!"),
]

response: AIMessage = llm.invoke(messages)

print("---------TYPE ANSWER---------")
print(type(response))

print("----------STR OUTPUT---------")
print(response.content)

print("------RESPONSE METADATA------")
for key, value in response.response_metadata.items():
    print(f"{key}: {value}")

print("--------USAGE METADATA---------")
for usage_data, value in response.usage_metadata.items():
    print(f"{usage_data}: {value}")


