from flask_restful import Resource, reqparse
from app.models.toxicity_detector import DetectorModel


class ToxicityDetector(Resource):
    def post(self):
        parser = DetectorModel.post_parser()
        data = parser.parse_args()
        preds = DetectorModel.get_prediction(data["text"])
        return preds, 200