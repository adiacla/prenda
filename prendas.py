
#Realizado por Alfredo Díaz
#Se entreno con el notebook  Clasificacion prendas con Python y Tensorflow - Redes Densas
#https://colab.research.google.com/drive/1QhtclhDH-jrmmEtnDGn8MCutliceAKGR?usp=sharing
#pip install streamlit-drawable-canvas
#python -m pip install -U scikit-image

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from io import BytesIO

model = None
emb_model = None

st.set_option('deprecation.showPyplotGlobalUse', False)
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def to_rgb(x):
    ''' Convertimos una imagen de escala de grises a RGB'''
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb.reshape(-1, 28, 28, 3)


def prepara_img(image_array):
    ''' preparamos una imagen para predecir el articulo de vestir '''
    img = cv2.resize(image_array, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(-1, 28, 28, 1)
    return to_rgb(img)

# =================================================================================
# CARGANDO Y PREDICIENDO LOS DIGITOS

@st.cache_resource
def load_model():
    ''' Cargando el modelo RED NEURONAL '''
    global model
    global emb_model
    if model is None or emb_model is None:
        model_file = open('modelo.json', 'r')
        model = model_file.read()
        model_file.close()
        model = tf.keras.models.model_from_json(model)
        model.load_weights('modelo.h5')

        emb_model = tf.keras.models.Model(model.input,
                                          model.get_layer('embedding').output)
    return model, emb_model


def predict_class(img):
    ''' Calculamos y graficamos las predicciones '''
    global model
    global emb_model
    articulo={0:'T-shirt/top', 1:'Trouser/Pants', 2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',
              6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    model, emb_model = load_model()
    predictions = model.predict(img)
    predictions = predictions.ravel()
    clase_predicha = int(predictions.argmax())
    prob_ = 100 * predictions[clase_predicha]
    if prob_>umbral:
        pred_df = pd.DataFrame(predictions, index=range(10))
        st.subheader(f'Clase predicha: {clase_predicha},  probabilidad: {prob_:.2f}')
        st.text("!Congratulations!")
        st.text("You could use this word: ")
        st.title(articulo[clase_predicha])
        st.text("La probabildiad de las opciones son: ")
        st.bar_chart(pred_df)
    else:
        st.text(f"No me atrevo a predecir, la tolerancia es menor a {umbral} vuelve a dibujar o cargar otra imagen")
        

# =================================================================================

st.title('                            ¡TRADUCE A INGLES UNA PRENDA DE VESTIR A PARTIR DE UN DIBUJO O UN ARCHIVO!')
st.markdown('''
                               La siguiente aplicación intenta traducir a inglés una prenda de vestir que usted dibuje. ''')
st.markdown('''                                     * Usamos redes neuronales convolucionales con Tensorflow y keras.''')

st.markdown('''                                                      ALFREDO DIAZ ''')

st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#2e22e6;" /> """, unsafe_allow_html=True)
st.Write(''' Nota: Solo fué entrenado con 9 categorías como: Camiseta/top, pantalones, Pull-over, Vestido, Abrigo/Saco, Sandalia, Camisa, Teniis, Bolso y Botas)

col1,col2,col3=st.columns(3)

with col2:
    umbral = st.slider('Seleccione el umbral de tolerancia ', 0, 100, 50,1)

tab1, tab2, tab3 = st.tabs(["Dibuja", "Carga Imagen", "Toma una foto"])



with tab1:
    st.write('''Dibuja la  prenda, debes rellenarla lo más posible, por ejemplo:''')
    st.image("https://complex-valued-neural-networks.readthedocs.io/en/latest/_images/code_examples_fashion_mnist_22_0.png", width=150)
    st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#ff5733;" /> """, unsafe_allow_html=True)
    #PAra dibujar a mano alzada.
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=15,
        stroke_color='#D3D3D3',
        background_color='#000000',
        width=400,
        height=400,
        drawing_mode='freedraw',
        key='canvas'
    )
    if canvas_result.image_data is not None:
        if st.button('PREDECIR DIBUJO',type="primary"):
                image_array = canvas_result.image_data.astype(np.uint8)
                st.image(image_array, caption='Imagen cargada')
                img = prepara_img(image_array=image_array)
                st.subheader('PREDICCIÓN')
                predict_class(img)
    
with tab2:
    st.markdown('''¡Carga una foto de la  prenda, de preferencia sin fondo, por ejemplo:''')
    st.image("https://m.media-amazon.com/images/I/41J5zICsnLL._SS40_.jpg", width=100)
    st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#ff5733;" /> """, unsafe_allow_html=True)
    #Para cargar fotos
    uploaded_file = st.file_uploader("Seleccione la imagen",type=['jpeg', 'png', 'jpg', 'webp'])
    if uploaded_file is not None:
        # Normaliza los datos de la imagen si no están en el rango [0.0, 1.0]
        #Los modelos de tf.keras son optimizados sobre batch o bloques, o coleciones de ejemplos por vez. De acuerdo a esto, aunque use una unica imagen toca agregarla a una lista:
    # Convertir la imagen a un objeto PIL
        file_bytes = uploaded_file.getvalue()
        # Convertir los datos en un objeto de tipo BytesIO
        image_stream = BytesIO(file_bytes)
        # Abrir la imagen con PIL
        image_pil = Image.open(image_stream)
        # Redimensionar la imagen a (28, 28)
        image_pil = ImageOps.invert(image_pil)
        #image_pil = ImageOps.invert(image_pil)
        # Convertir la imagen PIL a un array numpy
        imagen_array = np.array(image_pil, dtype=np.uint8)
        
         # Mostrar la imagen
        st.image(imagen_array, caption='Imagen cargada')
        img = prepara_img(image_array=imagen_array)

        
        if st.button('PREDECIR IMAGEN',type="primary"):
                st.subheader('PREDICCIÓN')
                predict_class(img)
    
with tab3:
    st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#ff5733;" /> """, unsafe_allow_html=True)
    st.title('''¡Toma una foto de la  prenda, de preferencia sin fondo''')
    st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#ff5733;" /> """, unsafe_allow_html=True)

    img_file_buffer = st.camera_input("Tomar foto de la prenda")


    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        image_stream = BytesIO(bytes_data)
        # Abrir la imagen con PIL
        image_pil = Image.open(image_stream)
        # edimensionar la imagen a (28, 28)
        image_pil = ImageOps.invert(image_pil)
        #image_pil = ImageOps.invert(image_pil)
        # Convertir la imagen PIL a un array numpy
        imagen_array = np.array(image_pil, dtype=np.uint8)
            
        # Mostrar la imagen
        st.image(imagen_array, caption='Foto cargada',width=300 )
        img = prepara_img(image_array=imagen_array)

            
        if st.button('PREDECIR FOTO',type="primary"):
            st.subheader('PREDICCIÓN')
            predict_class(img)
    st.markdown("""<hr style="height:5px;border:none;color:#ff5733;background-color:#ff5733;" /> """, unsafe_allow_html=True)
