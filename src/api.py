import logging
import os
import socket

import torch
from flask import Flask, jsonify, request
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel
from transformers import AutoModel

HOST_NAME = os.environ.get("APP_DNS", "localhost")
APP_NAME = os.environ.get("APP_NAME", "flask")
IP = os.environ.get("PYTHON_IP", "127.0.0.1")
PORT = int(os.environ.get("PYTHON_PORT", 8088))
HOME_DIR = os.environ.get("HOMEDIR", os.getcwd())

log = logging.getLogger(__name__)
app = Flask(__name__)


@app.route("/")
def hello():
    return jsonify(
        {
            "host_name": HOST_NAME,
            "app_name": APP_NAME,
            "ip": IP,
            "port": PORT,
            "home_dir": HOME_DIR,
            "host": socket.gethostname(),
        }
    )


def load_model():
    path = "./model/seq2seqmodel"
    model_loaded = AutoModel.from_pretrained(path, local_files_only=True)
    return model_loaded


def initialize_model(model_loaded):
    model_args = Seq2SeqArgs()
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        model=model_loaded,
        use_cuda=False,
    )
    return model


def get_model():
    global get_model
    model_loaded = load_model()
    model = initialize_model(model_loaded)

    def inner():
        return model

    get_model = inner
    return model


@app.route("/prediction", methods=["POST"])
def get_prediction():
    try:
        model = get_model()
        if request.method == "POST":
            to_analize = request.json.get("text")
            result = model.predict([to_analize])
            return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": e}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
