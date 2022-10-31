import pickle
import json
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class SentementClassifier(Resource):
    def __init__(self):
        pass

    def post(self):
        args = parser.parse_args()
        data = json.loads(args['data'])
        list = []
        for bit in data:
            bit = str(bit)
            texty = [bit]
            pred = model.predict(texty)
            list.append(int(pred[0])+1) # Json doesn't recognise np.int so setting to default int
        return jsonify(list)

api.add_resource(SentementClassifier, '/classifier')

if __name__ == '__main__':
    # Load model
    with open('model\\model.pickle', 'rb') as f:
        model = pickle.load(f)

    app.run(debug=True)