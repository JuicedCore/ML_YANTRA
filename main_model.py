from flask import Flask, request
from PIL import Image
from clip_interrogator import Config, Interrogator
import requests
from io import BytesIO
import base64

app = Flask(__name__)

image = Image.open("Mike.jpg")
caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

@app.route("/", methods=["GET","POST"])
def home():
    return "API is up and running"

@app.route("/image/upload", methods=["POST"])
def image_to_prompt():
    image_file = request.form["image"]
    image_file = base64.b64decode(image_file)
    image = Image.open(BytesIO(image_file))
    print(image)
    print(type(image))
    
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    a = ci.interrogate_fast(image)
    print(a, "HEY HOMEY")
    return a


if __name__ == "__main__":
    app.run(debug=True)
    image_to_prompt("Mike.jpg")
