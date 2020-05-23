import sys
import os

from flask import Flask, request, g
from flask_restful import Resource, Api
from app.resources.toxicity_detector import ToxicityDetector
from app.utils import load_models, load_vec, load_r

app = Flask(__name__)
api = Api(app)
api.add_resource(ToxicityDetector, "/toxicity_detectors")

@app.before_first_request
def load_models_():
    m = load_models('./saved_models')
    vec = load_vec("./data/train.csv")
    r = load_r("./saved_models")
    app.config['models'] = (m, vec, r)

if __name__ == "__main__":
    app.run(debug=True)