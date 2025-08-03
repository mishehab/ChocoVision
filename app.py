import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("chocolate_classifie.keras")

# Load the label metadata
labels_df = pd.read_csv("labels.csv")

# Create a label → metadata dictionary
metadata = {}
for _, row in labels_df.iterrows():
    metadata[row['label']] = {
        "price": row['price'],
        "manufacturer": row['manufacturer'],
        "calories": row['calories']
    }

# Set of unique class labels
class_names = sorted(labels_df['label'].unique())

# Prediction function
def predict(image):
    img = image.resize((224, 224)).convert('RGB')
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)[0]
    predicted_index = np.argmax(preds)
    predicted_label = class_names[predicted_index]
    return predicted_label, preds[predicted_index]

# Streamlit app layout
st.set_page_config(page_title="Chocolate Classifier", layout="centered")
st.title("🍫 Chocolate Category Classifier")
st.write("Take a picture of a chocolate bar to see its name, price, calories, and manufacturer.")

# Camera input
img_file = st.camera_input("📷 Capture a chocolate image")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Captured Image", use_column_width=True)

    predicted_label, confidence = predict(image)

    st.success(f"✅ Prediction: **{predicted_label}** ({confidence*100:.2f}% confidence)")

    info = metadata.get(predicted_label, {})
    st.write(f"💰 **Price**: {info.get('price', 'N/A')} BDT")
    st.write(f"🏭 **Manufacturer**: {info.get('manufacturer', 'N/A')}")
    st.write(f"🔥 **Calories**: {info.get('calories', 'N/A')} kcal")
