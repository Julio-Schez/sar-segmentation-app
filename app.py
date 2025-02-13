from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import folium
from folium.plugins import Draw
from PIL import Image
import numpy as np
import logging
import tifffile as tiff
import geopandas as gpd
import matplotlib.pyplot as plt
import io
from pathlib import Path
import tensorflow as tf
import warnings
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Conv2DTranspose, Dropout,
    Lambda, concatenate, Input
)
from tensorflow.keras.models import Model, load_model
import cv2
from skimage.transform import resize

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Inicialización de Flask
app = Flask(__name__)

# Configuración de carpetas y modelo
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads/')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed_images/')
MODEL_PATH = os.path.join(os.getcwd(), 'model/flood_model.keras')
TILE_SIZE = 512

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'shp', 'dbf', 'shx', 'prj'}

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Variables globales para almacenar las capas.
image_layers = []
shapefile_layers = []

class ShapefileGroup:
    required_extensions = {'.shp', '.dbf', '.shx', '.prj'}
    def __init__(self, base_name):
        self.base_name = base_name
        self.files = set()
    def add_file(self, filename):
        self.files.add(filename)
    def is_complete(self):
        return all(self.base_name + ext in self.files for ext in self.required_extensions)
    def get_shp_path(self):
        return os.path.join(UPLOAD_FOLDER, self.base_name + '.shp')

def pad_to_square(img_array):
    """Añade padding a la imagen para hacerla cuadrada."""
    height, width = img_array.shape[:2]
    target_size = max(height, width)
    if img_array.ndim == 3:
        pad_height = target_size - height
        pad_width = target_size - width
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        pad_height = target_size - height
        pad_width = target_size - width
        padding = ((0, pad_height), (0, pad_width))
    padded_img = np.pad(img_array, padding, mode='reflect')
    return padded_img, height, width

def preprocess_image(image_path):
    """Carga y preprocesa una imagen TIF para el modelo."""
    img = tiff.imread(image_path)

    # Si es una imagen monocanal, duplicarla
    if len(img.shape) == 2:
        img = np.stack((img, img), axis=-1)

    # Ajustar tamaño de la imagen
    img_resized = resize(img, (512, 512), mode='constant', anti_aliasing=True)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def convert_tif_to_png(tif_path, output_path):
    """Convierte una imagen TIFF a PNG para visualizarla en la web."""
    img = tiff.imread(tif_path)
    img = (img / np.max(img) * 255).astype(np.uint8)  # Normalizar valores a 0-255
    img_pil = Image.fromarray(img)  # Convertir a formato PIL
    img_pil.save(output_path, "PNG")

def allowed_file(filename):
    """
    Verifica si la extensión del archivo está permitida.
    Para archivos TIFF/TIF se requiere que el nombre contenga 'sentinel1' o 'sentinel-1'
    para asegurar que se procese la imagen correcta.
    """
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            if ext in ('tif', 'tiff'):
                if 'sentinel1' in filename.lower() or 'sentinel-1' in filename.lower():
                    return True
                else:
                    logging.warning(f"Archivo TIFF '{filename}' no contiene 'sentinel1' en el nombre.")
                    return False
            return True
    return False

def process_tiff_image(filepath, filename):
    """Procesa una imagen TIFF y devuelve la información necesaria para visualizarla."""
    try:
        image_array = tiff.imread(filepath).astype(np.float32)
        if image_array.ndim == 3 and image_array.shape[0] == 2:
            image_array = np.transpose(image_array, (1, 2, 0))
            logging.info(f"Transpuesta la imagen (process_tiff_image), nueva forma: {image_array.shape}")
        logging.info(f"Dimensiones de la imagen: {image_array.shape}")
        png_filename = filename.replace('.tif', '.png').replace('.tiff', '.png')
        png_path = os.path.join(app.config['PROCESSED_FOLDER'], png_filename)
        if image_array.ndim == 2:
            null_mask = np.isnan(image_array) | (image_array == 0)
        else:
            null_mask = np.isnan(image_array).any(axis=-1) | np.all(image_array == 0, axis=-1)
        image_array = np.nan_to_num(image_array, nan=0)
        image_min = image_array[~null_mask].min() if np.any(~null_mask) else 0
        image_max = image_array[~null_mask].max() if np.any(~null_mask) else 1
        scale = 255.0 / (image_max - image_min) if image_max > image_min else 1
        image_array = ((image_array - image_min) * scale).clip(0, 255).astype(np.uint8)
        if image_array.ndim == 2:
            cmap = plt.get_cmap('viridis')
            norm_image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            colored_image = cmap(norm_image)
            image_array = (colored_image[:, :, :3] * 255).astype(np.uint8)
        elif image_array.ndim == 3:
            if image_array.shape[2] == 1:
                cmap = plt.get_cmap('viridis')
                norm_image = (image_array[:, :, 0] - image_array.min()) / (image_array.max() - image_array.min())
                colored_image = cmap(norm_image)
                image_array = (colored_image[:, :, :3] * 255).astype(np.uint8)
            elif image_array.shape[2] == 2:
                image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)
            elif image_array.shape[2] >= 3:
                image_array = image_array[:, :, :3]
        alpha_channel = np.where(null_mask, 0, 255).astype(np.uint8)
        image_array = np.dstack((image_array, alpha_channel))
        image = Image.fromarray(image_array, 'RGBA')
        if os.path.exists(png_path):
            os.remove(png_path)
        image.save(png_path, format='PNG')
        logging.info(f"Imagen procesada guardada en {png_path}")
        height, width = image_array.shape[:2]
        try:
            with tiff.TiffFile(filepath) as tif:
                tags = tif.pages[0].tags
                metadata = {tag.name: tag.value for tag in tags.values()}
                model_pixel_scale = metadata.get('ModelPixelScaleTag')
                model_tiepoint = metadata.get('ModelTiepointTag')
                if model_pixel_scale and model_tiepoint and len(model_tiepoint) >= 6:
                    min_x = model_tiepoint[3]
                    min_y = model_tiepoint[4]
                    pixel_scale_x = model_pixel_scale[0]
                    pixel_scale_y = model_pixel_scale[1]
                    max_x = min_x + pixel_scale_x * width
                    max_y = min_y - pixel_scale_y * height
                    bounds = [[min_y, min_x], [max_y, max_x]]
                else:
                    raise ValueError("Metadatos insuficientes para calcular los límites geográficos.")
        except Exception as e:
            logging.warning(f"No se pudo extraer la georreferenciación: {e}")
            default_lat = 19.432608
            default_lon = -99.133209
            bounds = [[default_lat, default_lon], [default_lat + (height * 0.0001), default_lon + (width * 0.0001)]]
        return {'name': png_filename, 'bounds': bounds, 'is_mask': False, 'opacity': 0.7}  # Opacidad inicial
    except Exception as e:
        logging.error(f"Error procesando imagen TIFF: {str(e)}")
        raise

@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)

@app.route('/processed_images/<path:filename>')
def send_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/upload_shapefile', methods=['POST'])
def upload_shapefile():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud.'})
    try:
        files = request.files.getlist('file')
        if not files:
            return jsonify({'success': False, 'error': 'No se seleccionaron archivos.'})
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filename)
                logging.info(f"Archivo guardado: {filename}")
        shapefile_groups = group_shapefile_components(saved_files)
        for group in shapefile_groups.values():
            if group.is_complete():
                shp_path = group.get_shp_path()
                logging.info(f"Procesando shapefile: {shp_path}")
                base_name = os.path.basename(shp_path)
                if any(layer['name'] == base_name for layer in shapefile_layers):
                    continue
                shapefile = gpd.read_file(shp_path)
                bounds = shapefile.total_bounds
                bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
                shapefile_layers.append({
                    'name': base_name,
                    'bounds': bounds,
                    'layer': folium.GeoJson(shapefile, name=base_name)
                })
                logging.info(f"Shapefile procesado exitosamente: {base_name}")
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error en upload_shapefile: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        data = request.get_json()
        if not data or 'image_name' not in data:
            return jsonify({'success': False, 'error': 'No se proporcionó el nombre de la imagen.'})

        image_name = data['image_name']
        model = load_model(MODEL_PATH)

        # Buscar la imagen en las capas cargadas
        image_to_segment = None
        for img in image_layers:
            if img['name'] == image_name:
                image_to_segment = img
                break

        if not image_to_segment:
            return jsonify({'success': False, 'error': 'La imagen no se encontró en las capas cargadas.'})

        # Procesar la imagen seleccionada
        original_path = os.path.join(UPLOAD_FOLDER, image_to_segment['tiff'])
        logging.info(f"Segmentando imagen Sentinel-1: {original_path}")
        image = preprocess_image(original_path)
        prediction = model.predict(image)
        prediction = (prediction > 0.5).astype(np.uint8)

        # Crear una máscara azul
        blue_mask = np.zeros((prediction.shape[1], prediction.shape[2], 4), dtype=np.uint8)
        blue_mask[prediction[0, :, :, 0] == 1] = [0, 0, 255, 128]  # Azul con opacidad

        # Guardar la máscara azul como PNG
        png_mask = "mask_" + image_to_segment['tiff'].replace('.tif', '.png').replace('.tiff', '.png')
        png_mask_path = os.path.join(app.config['PROCESSED_FOLDER'], png_mask)
        Image.fromarray(blue_mask, 'RGBA').save(png_mask_path)

        # Agregar la máscara como una nueva capa sin eliminar la imagen original
        mask_info = {
            'name': png_mask,
            'bounds': image_to_segment['bounds'],
            'original': image_to_segment['tiff'],
            'is_mask': True,  # Marcar como máscara
            'opacity': 0.5  # Opacidad inicial para máscaras
        }
        image_layers.append(mask_info)  # <-- Agregar la máscara como nueva capa
        logging.info(f"Máscara de inundación generada: {png_mask}")

        return jsonify({'success': True, 'mask_name': png_mask})
    except Exception as e:
        logging.error(f"Error en la segmentación: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_layers, shapefile_layers
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud.'})
            files = request.files.getlist('file')
            if not files:
                return jsonify({'success': False, 'error': 'No se seleccionaron archivos.'})
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    logging.info(f"Archivo guardado: {filename}")
                    if filename.lower().endswith(('.tif', '.tiff')):
                        image_info = process_tiff_image(filepath, filename)
                        image_info['tiff'] = filename  # Guardamos el nombre original del TIFF
                        image_info['is_mask'] = False  # Marcar como imagen original
                        image_layers.append(image_info)
                        logging.info(f"Imagen TIFF procesada: {filename}")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error en la carga de archivos: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})
    default_lat = float(request.args.get('lat', 19.432608))
    default_lon = float(request.args.get('lon', -99.133209))
    folium_map = folium.Map(location=[default_lat, default_lon], zoom_start=5, control_scale=True)
    if image_layers or shapefile_layers:
        all_layers = image_layers + shapefile_layers
        if all_layers:
            first_layer = all_layers[0]
            bounds_all = [list(first_layer['bounds'][0]), list(first_layer['bounds'][1])]
            for layer in all_layers:
                bounds = layer['bounds']
                bounds_all[0][0] = min(bounds_all[0][0], bounds[0][0])
                bounds_all[0][1] = min(bounds_all[0][1], bounds[0][1])
                bounds_all[1][0] = max(bounds_all[1][0], bounds[1][0])
                bounds_all[1][1] = max(bounds_all[1][1], bounds[1][1])
            for image_layer in image_layers:
                is_mask = image_layer.get('is_mask', False)  # Verificar si es una máscara
                opacity = image_layer.get('opacity', 0.7)  # Usar la opacidad guardada
                display_name = "Máscara de Inundación" if is_mask else "Imagen Sentinel-1"
                overlay = folium.raster_layers.ImageOverlay(
                    image=url_for('send_processed_image', filename=image_layer['name'], _external=True),
                    bounds=image_layer['bounds'],
                    name=image_layer['name'],  # Nombre único para la capa
                    opacity=opacity,
                    overlay=True
                )
                overlay.add_to(folium_map)
                logging.info(f"Añadida capa {image_layer['name']} con bounds {image_layer['bounds']}")
            for shapefile_layer in shapefile_layers:
                shapefile_layer['layer'].add_to(folium_map)
            folium_map.fit_bounds(bounds_all)
    Draw(export=True).add_to(folium_map)
    folium.LayerControl(position='topleft').add_to(folium_map)
    return render_template('index.html', folium_map=folium_map._repr_html_(),
                           image_layers=image_layers, shapefile_layers=shapefile_layers,
                           alert_message=None)

@app.route('/clear', methods=['POST'])
def clear_map():
    try:
        global image_layers, shapefile_layers
        image_layers = []
        shapefile_layers = []
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error eliminando {file_path}: {str(e)}")
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error al limpiar el mapa: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
