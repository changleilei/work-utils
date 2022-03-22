# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/1/20 3:54 下午
==================================="""


from flask import Flask, jsonify
from server.blueprints.gpt3_blueprint import generate_blueprints
app = Flask(__name__)

app.register_blueprint(generate_blueprints, url_prefix="/generate")


@app.route("/")
def hello_world():
    return jsonify("Hello World!")


if __name__ == '__main__':
    app.run("0.0.0.0", port=8082)