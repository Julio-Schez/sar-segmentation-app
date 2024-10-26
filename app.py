from flask import Flask, render_template, request, redirect, url_for, send_from_directory
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

@app.route('/', methods=['GET', 'POST'])
def index():
    """Ruta principal que maneja la carga y visualización de archivos"""
    global image_layers, shapefile_layers

    # Configuración inicial del mapa
    default_lat = float(request.args.get('lat', 19.432608))
    default_lon = float(request.args.get('lon', -99.133209))
    folium_map = folium.Map(location=[default_lat, default_lon], 
                           zoom_start=5, 
                           control_scale=True, 
                           max_bounds=True)
    
    # Ajustar el mapa si hay capas existentes
    if image_layers or shapefile_layers:
        first_layer = image_layers[0] if image_layers else shapefile_layers[0]
        bounds_all = [list(first_layer['bounds'][0]), list(first_layer['bounds'][1])]
        for layer in image_layers + shapefile_layers:
            bounds_all[0][0] = min(bounds_all[0][0], layer['bounds'][0][0])
            bounds_all[0][1] = min(bounds_all[0][1], layer['bounds'][0][1])
            bounds_all[1][0] = max(bounds_all[1][0], layer['bounds'][1][0])
            bounds_all[1][1] = max(bounds_all[1][1], layer['bounds'][1][1])
        map_center = [(bounds_all[0][0] + bounds_all[1][0]) / 2, 
                     (bounds_all[0][1] + bounds_all[1][1]) / 2]
        folium_map.location = map_center
        folium_map.zoom_start = 12
        
        # Añadir capas de imagen existentes
        for image_layer in image_layers:
            overlay = folium.raster_layers.ImageOverlay(
                image=url_for('send_processed_image', filename=image_layer['name'], _external=True),
                bounds=image_layer['bounds'],
                name=image_layer['name'],
                opacity=0.7,
                interactive=True
            )
            overlay.add_to(folium_map)
        
        # Añadir capas de shapefile existentes
        for shapefile_layer in shapefile_layers:
            shapefile_layer['layer'].add_to(folium_map)
        
        folium_map.fit_bounds(bounds_all)
        folium_map.options['worldCopyJump'] = False
    
    # Añadir herramienta de dibujo
    draw = Draw()
    draw.add_to(folium_map)

    # Procesar archivos subidos
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.error("No se encontró el archivo en la solicitud.")
            return redirect(request.url)
        
        files = request.files.getlist('file')
        if files and all(allowed_file(f.filename) for f in files):
            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logging.info(f"Archivo guardado en {filepath}")
                
                try:
                    # Procesar shapefile
                    if file.filename.lower().endswith('.shp'):
                        shapefile = gpd.read_file(filepath)
                        bounds = shapefile.total_bounds  # [minx, miny, maxx, maxy]
                        bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
                        geojson_layer = folium.GeoJson(shapefile, name=filename, show=True)
                        geojson_layer.add_to(folium_map)
                        shapefile_layers.append({
                            'name': filename, 
                            'bounds': bounds, 
                            'layer': geojson_layer
                        })
                    
                    # Procesar imagen TIFF
                    elif file.filename.lower().endswith(('.tif', '.tiff')):
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
                            logging.warning(f"No se pudo extraer la georreferenciación: {e}. Utilizando límites aproximados.")
                            bounds = [[default_lat, default_lon], 
                                    [default_lat + (height * 0.0001), default_lon + (width * 0.0001)]]
                        
                        # Añadir capa al mapa
                        overlay = folium.raster_layers.ImageOverlay(
                            image=url_for('send_processed_image', filename=png_filename, _external=True),
                            bounds=bounds,
                            name=filename,
                            opacity=1.0
                        )
                        overlay.add_to(folium_map)
                        image_layers.append({'name': png_filename, 'bounds': bounds})

                except Exception as e:
                    logging.error(f"Error al procesar el archivo: {e}")
                    alert_message = f"Error al cargar el archivo {filename}: {str(e)}"
                    return render_template('index.html', 
                                        folium_map=folium_map._repr_html_(), 
                                        image_layers=image_layers, 
                                        alert_message=alert_message)
    
    # Añadir control de capas
    folium.LayerControl(position='topleft').add_to(folium_map)
    
    # Renderizar template
    map_html = folium_map._repr_html_()
    return render_template('index.html', 
                         folium_map=map_html, 
                         image_layers=image_layers, 
                         alert_message=None)

if __name__ == "__main__":
    app.run(debug=True)
