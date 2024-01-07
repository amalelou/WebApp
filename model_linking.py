from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline,ControlNetModel, DPMSolverMultistepScheduler
import accelerate
from PIL import Image
import torch
import transformers
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, send_file, request
import base64
from io import BytesIO
import cv2
from PIL import Image
import numpy as np

!ngrok authtoken '2aYSH1K18vRFqE66OEbV5lOMbRA_6JwrwwPRd7reFzBXYn2z1'

#Model importing
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", torch_dtype=torch.float32)

folder = '/content/drive/MyDrive/sd-concept-output'

pipe = StableDiffusionPipeline.from_pretrained(
    folder,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(folder, subfolder="scheduler"),
    torch_dtype=torch.float32,
).to('cuda')

pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
    folder,
    controlnet=controlnet,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(folder, subfolder="scheduler"),
    torch_dtype=torch.float32,
).to('cuda')

# Start flask app and set to ngrok
app = Flask(_name_, template_folder='.') # period is because default templates folder is /templates
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image1 = pipe(prompt).images[0]
  print("Image generated! Converting image ...")

  buffered1 = BytesIO()
  image1.save(buffered1, format="PNG")
  img_str1 = base64.b64encode(buffered1.getvalue())
  b1 = "data:image/png;base64," + str(img_str1)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image1=b1)

@app.route('/submit-prompt', methods=['POST'])
def generate_image1():
    prompt = request.form['prompt-input']
    print(f"Generating an image with prompt: {prompt}")

    uploaded_image = request.files['image-file']

    uploaded_image.save("uploaded_image.png")

    uploaded_image = np.array(uploaded_image)

    low_threshold = 100
    high_threshold = 200

    uploaded_image = cv2.Canny(uploaded_image, low_threshold, high_threshold)
    uploaded_image = uploaded_image[:, :, None]
    uploaded_image = np.concatenate([uploaded_image, uploaded_image, uploaded_image], axis=2)
    canny_image = Image.fromarray(uploaded_image)

    image2 = pipe1(prompt, canny_image).images[0]


    print("Image generated! Converting image ...")

    buffered2 = BytesIO()
    image2.save(buffered2, format="PNG")
    img_str2 = base64.b64encode(buffered2.getvalue())
    b2 = "data:image/png;base64," + str(img_str2)[2:-1]

    print("Sending image ...")
    return render_template('index.html', generated_image2=b2)

app.run()