from flask import Flask, request

from api import get_answer

app = Flask(__name__)

@app.route("/llm-api/")
def llm_api():
    return get_answer(request.args.get('q'))