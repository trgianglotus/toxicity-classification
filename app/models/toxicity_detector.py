from flask import g, current_app
from flask_restful import reqparse

import numpy as np

class DetectorModel():
    @classmethod
    def get_prediction(cls, text):
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        m, vec, r = current_app.config['models']
        preds = np.zeros((len(label_cols)))
        spare_matrix = vec.transform([text])
        for i in range(len(label_cols)):    
            preds[i] = m[i].predict_proba(spare_matrix.multiply(r[i]))[:,1]
        return DetectorModel.preds_to_json(preds, text)

    @staticmethod
    def preds_to_json(preds, text):
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        res = {}
        res['comment'] = text
        for i, j in enumerate(label_cols):
            res[j] = preds[i]
        return res

    @staticmethod
    def post_parser():
        parser = reqparse.RequestParser()

        parser.add_argument("text",
                            type=str,
                            required=True,
                            help="This field cannot be left blank.")

        return parser

