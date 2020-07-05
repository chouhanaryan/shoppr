from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

# from app import app
import os
import uuid
import json
import tensorflow as tf

from keras.preprocessing.image import img_to_array
from keras.applications import ResNet50, imagenet_utils

from PIL import Image
import numpy as np
import pickle
import flask
import io

from selectorlib import Extractor
import requests 
import json 
from time import sleep


# Create an Extractor by reading from the YAML file
e1 = Extractor.from_yaml_file('search_results.yml')
e2 = Extractor.from_yaml_file('selectors.yml')


ada_model = None
res_model = None
app = Flask(__name__)


@app.route("/prod")
def prod():
	return render_template("product_list.html")

@app.route("/")
@app.route("/testindex")
def testindex():
	return render_template("testindex.html")


# @app.route('/uploader', methods=['GET', 'POST'])
# def uploader():
#     if request.method == 'POST':
#         file = request.files['filename']
#         extension = os.path.splitext(file.filename)[1]
#         f_name = str(uuid.uuid4()) + extension
#         file.save(os.path.join('.', f_name))
#     return json.dumps({'filename': f_name})


def load_all_models():
	print("ENTER LOAD ALL MODELS")
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global ada_model, res_model

	with open("ada-vvlarge-964.pickle", "rb") as file:
		ada_model = pickle.load(file)
		print("ADA")

	print("RES")
	# res_model = load_model("resmodel.h5")

	json_file = open("model.json", "r")
	model_json = json_file.read()
	res_model = tf.keras.models.model_from_json(model_json)
	res_model.load_weights("resmodel.h5")

	print("EXIT LOAD ALL MODELS")


def prepare_image(image, target):
	print("ENTER PREPARE IMAGE")

	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	print("EXIT PREPARE IMAGE")

	# return the processed image
	return image


def scrape(url, e):
	headers = {
		'dnt': '1',
		'upgrade-insecure-requests': '1',
		'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
		'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
		'sec-fetch-site': 'same-origin',
		'sec-fetch-mode': 'navigate',
		'sec-fetch-user': '?1',
		'sec-fetch-dest': 'document',
		'referer': 'https://www.amazon.com/',
		'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
	}

	# Download the page using requests
	print("Downloading %s"%url)
	r = requests.get(url, headers=headers)
	# Simple check to check if page was blocked (Usually 503)
	if r.status_code > 500:
		if "To discuss automated access to Amazon data please contact" in r.text:
			print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
		else:
			print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
		return None
	# Pass the HTML of the page and create 
	return e.extract(r.text)


@app.route("/uploader", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# for i, val in enumerate

	classes = {
		i: val
		for i, val in enumerate(
			[
				"Shirts Navy Blue",
				"Shirts Blue",
				"Jeans Blue",
				"Sports Shoes Blue",
				"Shirts Black",
				"Sports Shoes Black",
				"Shirts Grey",
				"Sports Shoes Grey",
				"Shirts Green",
				"Shirts Purple",
				"Shirts White",
				"Sports Shoes White",
				"Shirts Red",
			]
		)
	}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image
			vec = res_model.predict(image)
			preds = ada_model.predict(vec)
			product = classes.get(int(preds[0]))
			url = 'https://www.amazon.com/s?k=' + product
			prod_list = []
			prod_list_url = []
			products = {}


			scraped = scrape(url, e1)
			print(scraped)
			if scraped:
				for i, product in enumerate(scraped['products']):
					if i == 5:
						break
					i += 1
					product['search_url'] = url
					print("Product: %s"%product['title'])
					prod_list.append(product['title'])
					prod_list_url.append("https://www.amazon.com"+product['url'])

			# data = {
			# 	str(i): val
			# 	for i, val in enumerate(prod_list)
			# }

			final_products = []

			for prod_url in prod_list_url:
				scraped_fin = scrape(prod_url, e2)				
				if scraped_fin:
					final_products.append({
						'name': scraped_fin.get('name', 'Not Available'),
						'price': scraped_fin.get('price', 'Not Available'),
						'short_description': scraped_fin.get('short_description', 'Not Available'),
						'image': scraped_fin['images'].strip('\n'),
						'url': prod_url
					})

			# indicate that the request was a success
			data["success"] = True
			data["products"] = final_products

	# return flask.jsonify(data)
	return render_template("product_list.html", data=data)


if __name__ == "__main__":
	print("Loading models...")
	load_all_models()
	app.run(debug=True)
