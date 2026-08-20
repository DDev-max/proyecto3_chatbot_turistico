# Chatbot de Reseñas Turísticas

Chatbot especializado en responder preguntas sobre atracciones turísticas a partir de reseñas reales, combinando recuperación de información (RAG), filtrado de relevancia mediante un modelo de clasificación afinado, y generación de respuestas con un modelo de lenguaje.

## Descripción general

El sistema procesa la pregunta del usuario en tres etapas:

1. **Recuperación (RAG):** se utiliza el modelo `multilingual-e5-small` para generar embeddings y extraer las reseñas más relevantes según la pregunta del usuario.
2. **Filtrado:** los chunks extraídos pasan por un modelo `DistilBERT` afinado (fine-tuning) que filtra y prioriza los fragmentos según su pertinencia.
3. **Generación de respuesta:** la pregunta junto con los chunks filtrados se envían a un modelo servido mediante Ollama, que genera la respuesta final siguiendo un conjunto de reglas estrictas definidas para el chatbot.

## Datos

Se utilizan más de 5.000 reseñas extraídas de Civitatis y Google Maps, clasificadas por categoría:

**Civitatis**

| Actividad | Categoría |
|---|---|
| Visita guiada por el Parque Güell | Aire libre |
| Visita guiada por el Museo del Prado y el Palacio Real | Históricos |
| Palacio Real + Catedral de la Almudena | Históricos |
| Entradas al Museo del Prado sin colas | Históricos |

**Google Maps**

| Actividad | Categoría |
|---|---|
| Volcán Arenal | Aire libre |
| Kalambu Hot Springs | Atracciones |
| Museo de los Niños Costa Rica | Atracciones |

### Polaridad

Cada reseña se clasifica según su polaridad con la siguiente nomenclatura:

| Etiqueta | Polaridad |
|---|---|
| LABEL_0 | Negativo |
| LABEL_1 | Neutro |
| LABEL_2 | Positivo |

## Estructura del proyecto

```
proyecto3_chatbot_turistico/
├── app/         # Chat interactivo en Plotly Dash
├── data/        # Archivos .csv con las reseñas
├── models/      # Embeddings, modelos y tensores
├── notebooks/   # Notebooks .ipynb
└── src/         # Funciones y utilidades del proyecto
```

## Requisitos

- Python 3.13.3
- [Ollama](https://ollama.com) instalado y en ejecución

### Modelo de lenguaje

Para equipos con recursos limitados se recomienda el modelo `qwen2.5:7b`, servido localmente mediante Ollama:

```
ollama run qwen2.5:7b
```

Requisitos orientativos para este modelo:

- 8 GB de RAM como mínimo (16 GB recomendado)
- 5 GB de espacio en disco para el modelo
- GPU opcional, mejora significativamente el tiempo de respuesta

Como alternativa, es posible usar la API de Gemini de forma gratuita, obteniendo una clave desde [Google AI Studio](https://aistudio.google.com/api-keys).

## Instalación

1. Clonar el repositorio.
2. Instalar las dependencias del proyecto (Python 3.13.3).
3. Instalar Ollama y descargar el modelo deseado, o configurar una API key de Gemini.

## Uso

Para ejecutar el chat interactivo:

```
python -m app.app
```

Esto inicia la aplicación desarrollada en Plotly Dash, donde el usuario puede realizar preguntas sobre las actividades turísticas disponibles y recibir respuestas basadas en las reseñas recopiladas.