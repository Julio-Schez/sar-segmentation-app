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
from tensorflow.keras.models import Model
import cv2
warnings.filterwarnings('ignore')


# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Inicialización de Flask
app = Flask(__name__)

# Configuración de carpetas y modelo
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads/')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed_images/')
MODEL_PATH = os.path.join(os.getcwd(), 'model/flood_model.h5')
TILE_SIZE = 512

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'shp', 'dbf', 'shx', 'prj'}

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Variables globales para almacenar las capas
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
    
    if len(img_array.shape) == 3:
        pad_height = target_size - height
        pad_width = target_size - width
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        pad_height = target_size - height
        pad_width = target_size - width
        padding = ((0, pad_height), (0, pad_width))
    
    padded_img = np.pad(img_array, padding, mode='reflect')
    return padded_img, height, width

class Sentinel1Processor:
    def __init__(self, model_path=MODEL_PATH):
        try:
            # Definir la arquitectura U-Net igual que en el entrenamiento
            def build_unet_model(input_size=(512, 512, 2)):
                inputs = Input(input_size)
                s = Lambda(lambda x: x / 255)(inputs)

                # Encoder
                c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
                c1 = Dropout(0.1)(c1)
                c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
                p1 = MaxPooling2D((2, 2))(c1)

                c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
                c2 = Dropout(0.1)(c2)
                c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
                p2 = MaxPooling2D((2, 2))(c2)

                c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
                c3 = Dropout(0.2)(c3)
                c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
                p3 = MaxPooling2D((2, 2))(c3)

                c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
                c4 = Dropout(0.2)(c4)
                c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
                p4 = MaxPooling2D((2, 2))(c4)

                c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
                c5 = Dropout(0.3)(c5)
                c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

                # Decoder
                u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
                u6 = concatenate([u6, c4])
                c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
                c6 = Dropout(0.2)(c6)
                c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

                u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
                u7 = concatenate([u7, c3])
                c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
                c7 = Dropout(0.2)(c7)
                c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

                u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
                u8 = concatenate([u8, c2])
                c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
                c8 = Dropout(0.1)(c8)
                c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

                u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
                u9 = concatenate([u9, c1], axis=3)
                c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
                c9 = Dropout(0.1)(c9)
                c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

                outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

                model = Model(inputs=[inputs], outputs=[outputs])
                return model

            # Crear el modelo con la arquitectura
            self.model = build_unet_model()
            
            # Cargar los pesos del modelo entrenado
            self.model.load_weights(model_path)
            logging.info("Modelo cargado exitosamente")
            
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
            
        self.tile_size = TILE_SIZE

    def preprocess_image(self, img_array):
        """Preprocesa la imagen para el modelo."""
        preprocessed = np.zeros_like(img_array, dtype=np.float32)
        for i in range(img_array.shape[-1]):
            band = img_array[..., i]
            min_val = np.percentile(band, 1)
            max_val = np.percentile(band, 99)
            preprocessed[..., i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
        return preprocessed

    def pad_image(self, img_array):
        """Añade padding a la imagen para que sea divisible por tile_size."""
        h, w, c = img_array.shape
        pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
        pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
        return np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    def create_tiles(self, img_array):
        """Divide la imagen en tiles."""
        h, w, c = img_array.shape
        tiles = []
        positions = []
        
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = img_array[y:y+self.tile_size, x:x+self.tile_size]
                if tile.shape[:2] == (self.tile_size, self.tile_size):
                    tiles.append(tile)
                    positions.append((y, x))
        
        return np.array(tiles), positions

    def reconstruct_image(self, predictions, positions, original_shape):
        """Reconstruye la imagen a partir de los tiles procesados."""
        h, w = original_shape[:2]
        reconstructed = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        for pred, (y, x) in zip(predictions, positions):
            reconstructed[y:y+self.tile_size, x:x+self.tile_size] += pred.squeeze()
            counts[y:y+self.tile_size, x:x+self.tile_size] += 1
        
        reconstructed = np.divide(reconstructed, counts, where=counts > 0)
        return reconstructed

    def process_image(self, input_path):
        """Procesa una imagen Sentinel-1 completa."""
        try:
            # Leer la imagen original usando tifffile
            image_array = tiff.imread(input_path)
            original_height, original_width = image_array.shape[:2]
            
            # Verificar que sea una imagen de dos bandas
            if len(image_array.shape) != 3 or image_array.shape[2] != 2:
                raise ValueError("La imagen debe tener exactamente 2 bandas (VV y VH)")

            # Hacer la imagen cuadrada con padding
            padded_img, orig_h, orig_w = pad_to_square(image_array)
            
            # Preprocesar imagen
            preprocessed = self.preprocess_image(padded_img)
            
            # Redimensionar a 512x512 si es necesario
            if preprocessed.shape[:2] != (512, 512):
                resized = np.zeros((512, 512, 2), dtype=preprocessed.dtype)
                for i in range(2):
                    resized[:, :, i] = cv2.resize(preprocessed[:, :, i], (512, 512))
                preprocessed = resized

            # Predecir
            prediction = self.model.predict(np.expand_dims(preprocessed, 0), verbose=0)[0]
            
            # Volver a las dimensiones originales
            if prediction.shape[:2] != (orig_h, orig_w):
                temp_mask = cv2.resize(prediction.squeeze(), (orig_w, orig_h))
                binary_mask = (temp_mask > 0.5).astype(np.uint8)
            else:
                binary_mask = (prediction.squeeze() > 0.5).astype(np.uint8)
            
            # Recortar al tamaño original
            binary_mask = binary_mask[:original_height, :original_width]

            # Crear una imagen RGB para la visualización
            visual_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
            # Áreas inundadas en azul semi-transparente
            visual_mask[binary_mask == 1] = [0, 0, 255]  # Azul
            
            # Convertir a RGBA para manejar transparencia
            rgba_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
            rgba_mask[..., :3] = visual_mask
            rgba_mask[..., 3] = np.where(binary_mask == 1, 180, 0)  # Alpha: 180 para áreas inundadas, 0 para el resto
            
            # Guardar máscara como PNG con transparencia
            png_output_path = input_path.replace('.tif', '_flood_mask.png')
            
            # Guardar usando PIL para mantener la transparencia
            Image.fromarray(rgba_mask, mode='RGBA').save(png_output_path)
            
            # Obtener georreferenciación de la imagen original
            try:
                with tiff.TiffFile(input_path) as tif:
                    tags = tif.pages[0].tags
                    metadata = {tag.name: tag.value for tag in tags.values()}
                    model_pixel_scale = metadata.get('ModelPixelScaleTag')
                    model_tiepoint = metadata.get('ModelTiepointTag')
                    
                    if model_pixel_scale and model_tiepoint and len(model_tiepoint) >= 6:
                        min_x = model_tiepoint[3]
                        min_y = model_tiepoint[4]
                        pixel_scale_x = model_pixel_scale[0]
                        pixel_scale_y = model_pixel_scale[1]
                        max_x = min_x + pixel_scale_x * original_width
                        max_y = min_y - pixel_scale_y * original_height
                        bounds = [[min_y, min_x], [max_y, max_x]]
                    else:
                        # Si no hay georreferenciación, usar la de la imagen original
                        original_info = next(layer for layer in image_layers 
                                          if layer['name'].replace('.png', '.tif') == os.path.basename(input_path))
                        bounds = original_info['bounds']
                        
            except Exception as e:
                logging.warning(f"Error al extraer georreferenciación: {str(e)}")
                # Usar bounds de la imagen original
                original_info = next(layer for layer in image_layers 
                                  if layer['name'].replace('.png', '.tif') == os.path.basename(input_path))
                bounds = original_info['bounds']
            
            logging.info(f"Máscara guardada con bounds: {bounds}")
            return png_output_path, bounds
                
        except Exception as e:
            logging.error(f"Error en el procesamiento de la imagen: {str(e)}")
            raise

def group_shapefile_components(filenames):
    """Agrupa los componentes del shapefile por su nombre base."""
    shapefile_groups = {}
    
    for filename in filenames:
        base_name = Path(filename).stem
        if '.' in base_name:  # Remove any remaining extensions
            base_name = base_name.rsplit('.', 1)[0]
            
        if base_name not in shapefile_groups:
            shapefile_groups[base_name] = ShapefileGroup(base_name)
        shapefile_groups[base_name].add_file(filename)
    
    return shapefile_groups

def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_tiff_image(filepath, filename):
    """Procesa una imagen TIFF y devuelve la información necesaria para visualizarla"""
    try:
        image_array = tiff.imread(filepath)
        logging.info(f"Dimensiones de la imagen: {image_array.shape}")
        png_filename = filename.replace('.tif', '.png').replace('.tiff', '.png')
        png_path = os.path.join(app.config['PROCESSED_FOLDER'], png_filename)

        # Identificar valores nulos o 0
        if len(image_array.shape) == 2:
            null_mask = np.isnan(image_array) | (image_array == 0)
        else:
            null_mask = np.isnan(image_array).any(axis=-1) | np.all(image_array == 0, axis=-1)

        # Reemplazar NaN por 0 para la normalización
        image_array = np.nan_to_num(image_array, nan=0)

        # Normalizar los valores a un rango de 0 a 255
        image_min = image_array[~null_mask].min() if np.any(~null_mask) else 0
        image_max = image_array[~null_mask].max() if np.any(~null_mask) else 1
        scale = 255.0 / (image_max - image_min) if image_max > image_min else 1
        image_array = ((image_array - image_min) * scale).clip(0, 255).astype(np.uint8)

        # Procesar imagen según el número de bandas
        if len(image_array.shape) == 2:
            cmap = plt.get_cmap('viridis')
            norm_image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            colored_image = cmap(norm_image)
            image_array = (colored_image[:, :, :3] * 255).astype(np.uint8)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                cmap = plt.get_cmap('viridis')
                norm_image = (image_array[:, :, 0] - image_array.min()) / (image_array.max() - image_array.min())
                colored_image = cmap(norm_image)
                image_array = (colored_image[:, :, :3] * 255).astype(np.uint8)
            elif image_array.shape[2] == 2:
                image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)
            elif image_array.shape[2] >= 3:
                image_array = image_array[:, :, :3]

        # Crear canal alpha basado en la máscara de valores nulos
        alpha_channel = np.where(null_mask, 0, 255).astype(np.uint8)
        
        # Combinar la imagen RGB con el canal alpha
        image_array = np.dstack((image_array, alpha_channel))
        
        # Guardar la imagen con transparencia
        image = Image.fromarray(image_array, 'RGBA')
        image.save(png_path, format='PNG')
        logging.info(f"Imagen procesada guardada en {png_path}")

        # Obtener georreferenciación
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
            bounds = [[default_lat, default_lon], 
                     [default_lat + (height * 0.0001), default_lon + (width * 0.0001)]]

        return {'name': png_filename, 'bounds': bounds}

    except Exception as e:
        logging.error(f"Error procesando imagen TIFF: {str(e)}")
        raise
@app.route('/static/<path:filename>')
def send_static(filename):
    """Ruta para servir archivos estáticos"""
    return send_from_directory('static', filename)

@app.route('/processed_images/<path:filename>')
def send_processed_image(filename):
    """Ruta para servir imágenes procesadas"""
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/upload_shapefile', methods=['POST'])
def upload_shapefile():
    """Endpoint para manejar la carga de shapefiles"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud.'})
    
    try:
        files = request.files.getlist('file')
        if not files:
            return jsonify({'success': False, 'error': 'No se seleccionaron archivos.'})

        # Guardar todos los archivos
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filename)
                logging.info(f"Archivo guardado: {filename}")

        # Agrupar los componentes del shapefile
        shapefile_groups = group_shapefile_components(saved_files)
        
        # Procesar shapefiles completos
        for group in shapefile_groups.values():
            if group.is_complete():
                shp_path = group.get_shp_path()
                logging.info(f"Procesando shapefile: {shp_path}")
                
                # Verificar si ya existe una capa con el mismo nombre
                base_name = os.path.basename(shp_path)
                if any(layer['name'] == base_name for layer in shapefile_layers):
                    continue
                
                shapefile = gpd.read_file(shp_path)
                bounds = shapefile.total_bounds  # [minx, miny, maxx, maxy]
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
    """Endpoint para realizar la segmentación de inundaciones"""
    try:
        # Verificar si hay imágenes Sentinel-1 cargadas
        sentinel_images = [layer for layer in image_layers 
                         if 'sentinel1' in layer['name'].lower() or 
                            'sentinel-1' in layer['name'].lower()]
        
        if not sentinel_images:
            return jsonify({
                'success': False, 
                'error': 'No se encontraron imágenes Sentinel-1 cargadas.'
            })
        
        processor = Sentinel1Processor()
        results = []
        
        for img in sentinel_images:
            # Obtener ruta de la imagen original
            original_path = os.path.join(
                UPLOAD_FOLDER, 
                img['name'].replace('.png', '.tif')
            )
            
            logging.info(f"Procesando imagen Sentinel-1: {original_path}")
            
            # Procesar imagen y obtener la máscara con sus bounds
            mask_path, bounds = processor.process_image(original_path)
            
            # Crear información de la capa de máscara
            mask_info = {
                'name': os.path.basename(mask_path),
                'bounds': bounds
            }
            
            image_layers.append(mask_info)
            results.append(mask_info['name'])
            
            logging.info(f"Máscara de inundación generada: {mask_info['name']}")
        
        return jsonify({
            'success': True,
            'processed_images': results
        })
        
    except Exception as e:
        logging.error(f"Error en la segmentación: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/', methods=['GET', 'POST'])
def index():
    """Ruta principal que maneja la carga y visualización de archivos"""
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
                        image_layers.append(image_info)
                        logging.info(f"Imagen TIFF procesada: {filename}")
            
            return jsonify({'success': True})
            
        except Exception as e:
            logging.error(f"Error en la carga de archivos: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})
    
    # Configuración inicial del mapa
    default_lat = float(request.args.get('lat', 19.432608))
    default_lon = float(request.args.get('lon', -99.133209))
    folium_map = folium.Map(location=[default_lat, default_lon], 
                           zoom_start=5, 
                           control_scale=True)
    
    # Ajustar el mapa si hay capas existentes
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
            
            # Añadir capas de imagen
            for image_layer in image_layers:
                # Determinar la opacidad y el nombre según si es una máscara o no
                is_mask = 'mask' in image_layer['name'].lower()
                opacity = 0.5 if is_mask else 0.7
                display_name = "Máscara de Inundación" if is_mask else "Imagen Sentinel-1"

                overlay = folium.raster_layers.ImageOverlay(
                    image=url_for('send_processed_image', filename=image_layer['name'], _external=True),
                    bounds=image_layer['bounds'],
                    name=display_name,
                    opacity=opacity,
                    overlay=True
                )
                overlay.add_to(folium_map)
                logging.info(f"Añadida capa {image_layer['name']} con bounds {image_layer['bounds']}")
            
            # Añadir capas de shapefile
            for shapefile_layer in shapefile_layers:
                shapefile_layer['layer'].add_to(folium_map)
            
            folium_map.fit_bounds(bounds_all)
    
    # Añadir herramienta de dibujo y control de capas
    Draw(export=True).add_to(folium_map)
    folium.LayerControl(position='topleft').add_to(folium_map)
    
    return render_template('index.html', 
                         folium_map=folium_map._repr_html_(), 
                         image_layers=image_layers,
                         shapefile_layers=shapefile_layers,
                         alert_message=None)

@app.route('/clear', methods=['POST'])
def clear_map():
    """Endpoint para limpiar el mapa"""
    try:
        global image_layers, shapefile_layers
        
        # Limpiar las listas de capas
        image_layers = []
        shapefile_layers = []
        
        # Limpiar archivos de las carpetas
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
    app.run(debug=True)
