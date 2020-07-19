from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin# runs the server globally
from com_in_ineuron_ai_utils.utils import decodeImage#customised package
from predict import FamilyClassfier

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)




#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "20180406_203222.jpg"
        self.classifier = FamilyClassfier(self.filename)



@app.route("/", methods=['GET'])#pass parameter as URL
@cross_origin()
def home():
    return render_template('index.html')
    


@app.route("/predict", methods=['POST'])#where to launch HTML page
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictionfamily()
    return jsonify(result)


port = int(os.getenv("PORT")) #for Cloud deployment -commented to check on local
#push application on cloud
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=port)#for local shoudl be 5000
    #app.run(host='0.0.0.0', port=5000, debug=True) #port number# to run locally
