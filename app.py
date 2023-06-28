import flask
import os
from flask import request
import uuid
import cv2
from flask import Flask,jsonify
from search_face import SearchFace
from inference import Inference
app = Flask(__name__)



inference = Inference()
faceSearch = SearchFace()
@app.route('/upload/<mahocsinh>', methods=['GET', 'POST'])
def upload(mahocsinh=None):
    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        path = 'faces/'+f_name
        file.save(path)
           
        image = cv2.imread('faces/'+f_name)
        emb = inference.inference(image)
        faceSearch.save_file(emb,mahocsinh)
        print('ok')
        return jsonify('ok')
    return 'err'
@app.route('/face_reg',methods = ['GET','POST'])
def search_face():
    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        path = 'faces/'+f_name
        file.save(path)
        image = cv2.imread('faces/'+f_name)
        emb = inference.inference(image)
        id = faceSearch.search(emb)
        print(id)
        return jsonify(str(id))
    return 'err'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)
