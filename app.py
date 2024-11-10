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

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Inicialización de Flask
app = Flask(__name__)

# Configuración de carpetas
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads/')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed_images/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'shp', 'dbf', 'shx', 'prj'}

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

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

@app.route('/static/<path:filename>')
def send_static(filename):
    """Ruta para servir archivos estáticos"""
    return send_from_directory('static', filename)

@app.route('/processed_images/<path:filename>')
def send_processed_image(filename):
    """Ruta para servir imágenes procesadas"""
    return send_from_directory(PROCESSED_FOLDER, filename)

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
        bounds = None
        
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
        except Exception as e:
            logging.warning(f"No se pudo extraer la georreferenciación: {e}")
        
        if bounds is None:
            # Usar límites aproximados si no se puede obtener la georreferenciación
            default_lat = 19.432608
            default_lon = -99.133209
            bounds = [[default_lat, default_lon], 
                     [default_lat + (height * 0.0001), default_lon + (width * 0.0001)]]

        return {'name': png_filename, 'bounds': bounds}

    except Exception as e:
        logging.error(f"Error procesando imagen TIFF: {str(e)}")
        raise

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
    
    # Configuración del mapa para GET request
    default_lat = float(request.args.get('lat', 19.432608))
    default_lon = float(request.args.get('lon', -99.133209))
    folium_map = folium.Map(location=[default_lat, default_lon], 
                           zoom_start=5, 
                           control_scale=True, 
                           max_bounds=True)
    
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
                overlay = folium.raster_layers.ImageOverlay(
                    image=url_for('send_processed_image', filename=image_layer['name'], _external=True),
                    bounds=image_layer['bounds'],
                    name=image_layer['name'],
                    opacity=0.7
                )
                overlay.add_to(folium_map)
            
            # Añadir capas de shapefile
            for shapefile_layer in shapefile_layers:
                shapefile_layer['layer'].add_to(folium_map)
            
            # Ajustar vista del mapa
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
