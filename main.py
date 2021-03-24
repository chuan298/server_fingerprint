import cv2, os, io, base64, jwt, json, time, requests
from datetime import datetime, timedelta, date
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file, Response, session, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
from gevent.pywsgi import WSGIServer
from time import strftime
import pymongo
from bson.objectid import ObjectId
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '27017')
DB_DATABASE = os.environ.get('DB_DATABASE', 'fingerprint')

# config server
# app = Flask(__name__)
app = Flask(__name__, template_folder='home')
CORS(app)

MONGO_URI = 'mongodb://' + DB_HOST + ':' + DB_PORT + '/'
myclient = pymongo.MongoClient(MONGO_URI)
mydb = myclient[DB_DATABASE]

mycol = mydb['data']


@app.route("/")
def index():
    return "Hello"


@app.route('/api/add_fingerprint', methods=['POST'])
def insert():
    print("PROCESS INSERT FINGERPRINT", flush=True)
    try:

        request_json = request.get_json()
        id_company = request_json.get('id_company', '')
        fingerprint = request_json.get('fingerprint', '')
        print("id_company", id_company)
        print("fingerprint", fingerprint)
        if id_company == "" or fingerprint == "":
            return {"result_code": 500}
        name, fp = fingerprint.split("@@@")

        data = {
            "name": name,
            "value": fp
        }
        # result = mycol.find_one({"id_company": ObjectId(id_company)})

        x = mycol.update({'id_company': id_company}, {'$push': {'fingerprints': data}})
        print(x)
        return {"result_code": 200}

    except Exception as e:
        print(e, flush=True)
    return jsonify({"result_code": 500})


@app.route('/api/get_fingerprint/<id_company>', methods=['GET'])
def get_all(id_company):
    print("PROCESS GET FINGERPRINTS", flush=True)
    try:
        if id_company == "":
            return {"result_code": 500}

        result = mycol.find({"id_company": id_company})

        data = ""
        for jData in result:
            data = jData['fingerprints']
        print({"result_code": 200, 'fingerprints': data})
        return jsonify({"result_code": 200, 'fingerprints': data, "id_company": id_company})

    except Exception as e:
        print(e, flush=True)
    return {"result_code": 500}

@app.route('/api/verify', methods=['POST'])
def verify():
    print("PROCESS VERIFY", flush=True)
    try:

        request_json = request.get_json()
        fingerprint1 = request_json.get('fingerprint1', '')
        fingerprint2 = request_json.get('fingerprint2', '')

        if fingerprint1 == "" or fingerprint2 == "":
            return {"result_code": 500}

        fingerprint1 = list(bytes(base64.b64decode(fingerprint1)))
        fingerprint2 = list(bytes(base64.b64decode(fingerprint2)))
        if len(fingerprint1) > len(fingerprint2):
            indices = [i for i, x in enumerate(fingerprint1) if x == 0]
            for i in range(len(fingerprint1) - len(fingerprint2)):
                # b.insert(indices[i] if len(indices) > i else -1, 0)
                fingerprint2.insert(-1, 0)

        else:
            indices = [i for i, x in enumerate(fingerprint2) if x == 0]
            for i in range(len(fingerprint2) - len(fingerprint1)):
                # a.insert(indices[i] if len(indices) > i else -1, 0)
                fingerprint1.insert(-1, 0)

        distance = np.linalg.norm(np.asarray(fingerprint1) - np.asarray(fingerprint2))
        # result = mycol.find_one({"id_company": ObjectId(id_company)})

        # x = mycol.update({'id_company': id_company}, {'$push': {'fingerprints': data}})
        # print(x)
        return {"result_code": 200, "distance": distance}

    except Exception as e:
        print(e, flush=True)
    return jsonify({"result_code": 500})

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 1234), app)
    http_server.serve_forever()