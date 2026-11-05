import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la página web
st.title("🐾 Clasificador de Perros y Gatos")
st.write("Proyecto parcial de Programación - ENES Unidad León")
st.markdown("""
**Trabajo de:**  
Mónica López  
César Herrera  
Christopher Mondragón  
""")
st.write("Sube una imagen de un perro o un gato para que la red neuronal profunda prediga el resultado.")

# Cargar el modelo entrenado
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('modelo_perros_gatos.keras')
    return model

with st.spinner('Cargando el modelo de inteligencia artificial...'):
    model = load_my_model()

uploaded_file = st.file_uploader("Elige una imagen (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen seleccionada en la pantalla
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada por el usuario')
    
    st.write("Analizando la imagen con la red neuronal...")
    

    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0  # Normalizar píxeles
    img_array = np.expand_dims(img_array, axis=0) # Ajustar dimensiones para el lote
    
  
    prediction = model.predict(img_array)[0][0]
    
    # Mostrar el resultado
    if prediction > 0.5:
        st.success(f"🐶 ¡Es un **Perro**! (Confianza: {prediction * 100:.2f}%)")
    else:
        st.error(f"🐱 ¡Es un **Gato**! (Confianza: {(1 - prediction) * 100:.2f}%)")