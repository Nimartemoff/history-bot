import requests

url = "http://localhost:5000/llm-api"


def get_answer(question):
    response = requests.get(url, params={'q': question})

    if response.status_code == 200:
        return response.text
