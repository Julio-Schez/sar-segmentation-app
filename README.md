# SAR-Segment-Flask: Plataforma Web para Visualización y Segmentación de Imágenes SAR

## 📋 Descripción
SAR-Segment-Flask es una aplicación web desarrollada con Flask que permite la visualización y análisis de imágenes satelitales, con énfasis en la detección y segmentación de inundaciones utilizando imágenes SAR de Sentinel-1. La plataforma también soporta la visualización de imágenes ópticas de Sentinel-2 y la integración de archivos vectoriales (SHP).

## 🚀 Características Principales
- Carga y visualización de imágenes satelitales:
  - Sentinel-1 (SAR)
  - Sentinel-2 (Óptico)
- Soporte para formatos:
  - Imágenes: `.tif`, `.tiff`
  - Vectorial: `.shp, .dbf, .shx, .prj`
- Segmentación automática de inundaciones en imágenes SAR
- Visualización interactiva de resultados
- Integración de capas vectoriales

## 💻 Requisitos del Sistema
```
Python 3.8+
Flask
GDAL
Rasterio
Geopandas
NumPy
(otros requisitos específicos por definir)
```

## 🔧 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/SAR-Segment-Flask.git
cd SAR-Segment-Flask
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🎯 Uso

1. Iniciar la aplicación:
```bash
python app.py
```

2. Abrir el navegador web y acceder a:
```
http://localhost:5000
```

3. Cargar imágenes:
   - Seleccionar archivo TIF/TIFF de Sentinel-1 o Sentinel-2
   - Opcionalmente, cargar archivo SHP para visualización de capas vectoriales

4. Utilizar las herramientas de segmentación para detectar inundaciones

## 🗺️ Ejemplos de Uso
[Aquí se pueden incluir capturas de pantalla o GIFs mostrando la funcionalidad]

## 🤝 Contribución
Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el proyecto
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📝 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE.md](LICENSE.md) para más detalles.

## 📞 Contacto
- Nombre del desarrollador
- Email: ejemplo@email.com
- GitHub: [@usuario](https://github.com/usuario)

## 🙏 Agradecimientos
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) por proporcionar acceso a las imágenes Sentinel
- Comunidad de desarrollo de herramientas geoespaciales
