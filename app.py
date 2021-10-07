from flask import Flask

from fastai import load_learner
from fastai.vision import open_image
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


@app.route("/")
def home():
    return "Hello World"


@app.route("/pred", methods=['POST'])
def predict():
    file = request.files['file']
    file.save(file.filename)
    img = open_image(file.filename)
    learn = load_learner(".")
    k = str(learn.predict(img))
    return k


if __name__ == '__main__':
    app.debug = True
    app.run()
