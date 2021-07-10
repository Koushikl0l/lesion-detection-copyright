from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions,preprocess_input



app = Flask(__name__)

model = ResNet50(include_top=True,weights='imagenet')

#model.make_predict_function()

def predict_label(img_path):
	img =image.load_img(img_path, target_size=(224,224))
	img = img_to_array(img)
	img = img.reshape(1, 224,224,3)
	img=preprocess_input(img)
	yhat=model.predict(img)
	label=decode_predictions(yhat)
	label=label[0][0]
	classification='%s (%.2f%%)' % (label[1],label[2]*100)
	return classification


# routes
@app.route("/",methods = ['GET'])
def main():
	return render_template("index.html")



@app.route("/submit",methods = ['GET','POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "./static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(port=3000,debug = True)
