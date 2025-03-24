import base64
import os
from mimetypes import guess_type
import streamlit as st
from openai import AzureOpenAI

class ImageClassificator:
    def __init__(self):
        # Configuración de la API
        self.api_base = st.secrets["AZURE_OAI_ENDPOINT"]
        self.api_key = st.secrets["AZURE_OAI_KEY"]
        self.deployment_name = st.secrets["AZURE_OAI_DEPLOYMENT"]
        self.api_version = "2024-02-15-preview"

        # Inicializar el cliente de Azure OpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )

        # Rutas predefinidas para las imágenes de ejemplo
        self.imagen_texto_DSCN = os.path.join('ImagenesEntrenamiento', 'DSCN7183.jpg')
        self.imagen_texto_1 = os.path.join("ImagenesEntrenamiento", "refri1.jpg")
        self.imagen_texto_4 = os.path.join("ImagenesEntrenamiento", "refri4.jpg")

        self.organizado = os.path.join('ImagenesEntrenamiento', 'Organizado.jpg')
        self.intermedio = os.path.join('ImagenesEntrenamiento', 'Intermedio.jpg')
        self.desorganizado = os.path.join('ImagenesEntrenamiento', 'Desorganizado.jpg')

    def local_image_to_data_url(self, image_path):
        """Codifica una imagen local en formato de data URL."""
        # Verificar si el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró el archivo: {image_path}")

        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def clasificar_pasillo(self, imagen_evaluar_path):
        """Clasifica la imagen del pasillo en función de los ejemplos proporcionados."""
        # Codificar las imágenes en formato de data URL
        imagen_texto_DSCN_url = self.local_image_to_data_url(self.imagen_texto_DSCN)
        imagen_texto_1_url = self.local_image_to_data_url(self.imagen_texto_1)
        imagen_texto_4_url = self.local_image_to_data_url(self.imagen_texto_4)


        organizado_url = self.local_image_to_data_url(self.organizado)
        intermedio_url = self.local_image_to_data_url(self.intermedio)
        desorganizado_url = self.local_image_to_data_url(self.desorganizado)


        imagen_evaluar_url = self.local_image_to_data_url(imagen_evaluar_path)

        # Crear la lista de mensajes para enviar a la API
        messages = [
            { "role": "system", "content": """
             Tus objetivos son 
             1. Decirme todas las marcas que ves en un refrigerador. 
                Para ellos se te poveeran de ejemplos 
             2. Clasificar si el refrigerador está organizado, medianamente orgnizado o desorganizado basado en cómo están los productos organizados (lácteos hasta arriba y embutidos abajo)
                Para ellos se te proveerán de imagenes de base para que puedas saber a qué se refiere cada clase.
                Tipo de clasificación multiclase:
                * Organizado
                * Medianamente organizado
                * Desorganizado
             
             Tu objetivo es aprender bien de los ejemplos y extarer el texto y dar una clasificación del neuvo refrigerador.
             
             """ },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un refrigerador ordenado ya que NO tiene huecos sin llenar de productos, y aun más importante los productos similares se encuentran cercanos, la leche con la leche, las verduras juntas, y las frutas en otro lado juntas. Es atractivo para los clientes." },
                    { "type": "image_url", "image_url": { "url": organizado_url } }
                ]
            },
            # {
            #     "role": "user",
            #     "content": [
            #         { "type": "text", "text": "Este es un refrigerador medianamente ya que hay huecos entre productos sin llenar, aunque sí cumple en cuanto a la cercanía entre productos similares sin combinar de diferentes tipos."},
            #         { "type": "image_url", "image_url": { "url": intermedio_url } }
            #     ]
            # },
            # {
            #     "role": "user",
            #     "content": [
            #         { "type": "text", "text": "Este es un refrigerador desorganizado ya que los productos están por todos lados, algunos acostados, otros parados y no es atractivo para los clientes. Además, algunas marcas se combinan entre ellas, agrupándose unas entre otras." },
            #         { "type": "image_url", "image_url": { "url": desorganizado_url } }
            #     ]
            # },
            # {
            #     "role": "user",
            #     "content": [
            #         { "type": "text", "text": "En este refrigerador se pueden apreciar marcas como 'Hortex', 'Wisnie', 'Truskawki', 'Maliny'. El refrigerador está medianamente organizado ya que se tienen algunos producto como el de 'La villita´  están aplastado junto con los de Yoplait."},
            #         { "type": "image_url", "image_url": { "url": imagen_texto_DSCN_url } }
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "En este refrigerador se pueden apreciar marcas como 'Yoplait', 'Fud', 'La villita', 'Olé'. Este refrigerador está medianamente organizado aunque ya que hay que considerar que no está bien abastecido."},
                    { "type": "image_url", "image_url": { "url": imagen_texto_1_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "En este refrigerador se pueden apreciar marcas como 'Yoplait', 'Fud', 'Olé', 'Queso Philadelphia'. Este refrigerador está mal organizado ya que los lácteos están por todas partes y las embutidos en medio."},
                    { "type": "image_url", "image_url": { "url": imagen_texto_4_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": """
                    Basándote en los ejemplos anteriores, clasifica la siguiente imagen según su nivel de organización. Asegúrate de que las recomendaciones estén orientadas específicamente para un **supermercado**.

                    El formato será el siguiente:

                    **Decisión**: (Selecciona entre 'Organizado', 'Medianamente organizado' o 'Desorganizado')
                     
                    **Marcas detectadas* [Lista de todas las marcas detectadas en formato de markdown]:
                        1. Marca a
                        2. Marca b
                        ...
                        3. Marca n

                    **Descripción**: (Describe brevemente qué ves en la imagen y qué elementos destacan. Sé claro y directo.)

                    **Recomendación**: (Proporciona una recomendación específica y detallada para mejorar la organización, enfocándote en un entorno de supermercado. Usa nombres, colores, marcas, estilos o cualquier detalle relevante. Sé breve pero útil, y evita información innecesaria. Por ejemplo:  
                    - "Noto que las botellas de Coca-Cola están mezcladas con otras marcas. Sugiero agrupar todas las botellas de Coca-Cola juntas, alineadas en la parte superior del estante, mientras que las botellas de Pepsi podrían estar en la fila inferior para diferenciarlas claramente por marca."  
                    - "Veo bolsas de frutos rojos mezcladas con bolsas de verduras verdes. Recomiendo colocar las bolsas de frutos rojos —como las que tienen etiquetas rojas— en el lado izquierdo del estante, y las bolsas de verduras en el lado derecho para separar las categorías."  
                     
 
                   IMPORTANTE: considera que un refrigerador debe de estar ordenado asi:
                    
                    En un refrigerador de tienda, la organización debe enfocarse en la exhibición atractiva, funcionalidad y facilidad de acceso para los clientes, cumpliendo además con normativas de seguridad alimentaria.

                    1. Orden por Categoría:
                    • Lácteos (leche, quesos, cremas, yogures): Parte superior o media del refrigerador, donde la temperatura es más estable.
                    • Embutidos y carnes frías (jamón, salchichas, chorizos, tocino): Cajones inferiores o zonas específicas para carnes frías, sellados para evitar derrames.
                    • Otros productos empacados de Sigma Alimentos (salsas, dips, etc.): Parte media o lateral del refrigerador, asegurando visibilidad y accesibilidad.
                    2. Orden por Fecha de Caducidad:
                    • Colocar los productos con fechas de caducidad más próximas al frente.
                    • Los productos más frescos deben ubicarse al fondo.
                    3. Temperatura Ideal:
                    • Embutidos y quesos blandos deben estar en las zonas de frío controlado (parte inferior o media).
                    • Lácteos (leche y yogures) deben mantenerse en la parte media o superior, donde la temperatura sea constante.
                    4. Prevención de Contaminación Cruzada:
                    • Asegurar que los productos estén sellados y no entren en contacto con otros alimentos crudos o abiertos.
                    • Los productos de Sigma Alimentos deben mantenerse separados de alimentos que no pertenecen a la misma categoría (e.g., productos crudos o de otras marcas).
                    5. Accesibilidad:
                    • Los productos más utilizados deben estar al frente o a la altura de los ojos para facilitar su acceso.
                    
                     
                    Por lo tanto, dí cómo deben de estar organizados los productos que ves con base en esa información de manera explícita. Menciona el porqué deben de estar en una sección u otra.
                     
                     NO menciones nada del estante, siempre menciona sobre el refrigerador.

                    Si todo está bien organizado, en **Recomendación** escribe: 'No hay recomendación, ¡todo está perfecto!'

                    **Nota**: Recuerda que esta evaluación está diseñada para un supermercado, por lo que las recomendaciones deben ser prácticas para este contexto. Por ejemplo, prioriza la agrupación por marca o categoría de producto en lugar de características como tamaño o color, ya que esto es más relevante para un cliente en este entorno.
                     
                    **Importante**: NO inventes marcas, únicamente menciona las que veas y si no alcanzas a ver o no estás seguro, no inventes.

                    """ },
                    { "type": "image_url", "image_url": { "url": imagen_evaluar_url } }
                ]
            }
        ]

        # Enviar la solicitud a la API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=1_500,
            temperature=0.1
        )

        # Devolver solo el contenido de la respuesta
        return response.choices[0].message.content


# # Ejemplo de uso
# clasificador = ImageClassificator()
# resultado = clasificador.clasificar_pasillo(imagen_evaluar_path=r'C:\Users\EmilioSandovalPalomi\OneDrive - Mobiik\Documents\Sigma\Demo_Refrigeradores_evento_sigma\ImagenesSigma\refri2.jpg')
# print(resultado)
