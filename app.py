import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Your class labels
class_names = [
    "battery", "keyboard", "microwave", "mobile", "mouse",
    "pcb", "player", "printer", "television", "washing machine"
]

# Prediction function
def classify_image(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return class_names[np.argmax(prediction)]

# Gradio UI
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

iface.launch()