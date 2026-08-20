# python -m app/app.py
import dash
from dash import dcc, html, Input, Output, State
from sentence_transformers import SentenceTransformer
from src.Filtrador_finetuned import Filtrador_finetuned
from src.busqueda_faiss import buscar_en_estrategia
from src.responder_ia import responder_ia


# ====================================================================
# 1. CARGA DE MODELOS (una sola vez, al iniciar el servidor)
# ====================================================================
print("Cargando modelo de embeddings (intfloat/multilingual-e5-small)...")
modelo_e5 = SentenceTransformer('intfloat/multilingual-e5-small')

print("Cargando filtrador fine-tuned...")
filtrador = Filtrador_finetuned()

MODELOS_LOCALES = ['qwen2.5:7b', 'qwen2.5:3b', 'gemma2:2b', 'llama3.2:3b']
MODELO_GEMINI = 'gemini-3.5-flash-lite'

OPCIONES_MODELO = [
    {'label': f' {m}  (local)', 'value': m} for m in MODELOS_LOCALES
] + [
    {'label': f' {MODELO_GEMINI}  (API Gemini)', 'value': MODELO_GEMINI}
]


# ====================================================================
# 2. HELPERS DE RENDERIZADO
# ====================================================================
def burbuja_mensaje(rol, texto):
    """Crea una burbuja de chat alineada según quién habla.

    rol debe ser 'user' o 'bot'.
    """
    clase = 'burbuja-usuario' if rol == 'user' else 'burbuja-bot'
    alineacion = 'flex-end' if rol == 'user' else 'flex-start'
    etiqueta = 'Tú' if rol == 'user' else 'Asistente'
    return html.Div(
        style={'display': 'flex', 'justifyContent': alineacion, 'margin': '10px 0'},
        children=html.Div(
            className=clase,
            children=[
                html.Div(etiqueta, className='etiqueta-burbuja'),
                html.Div(texto, className='texto-burbuja'),
            ],
        ),
    )


def render_contexto(contexto_metadata):
    """Renderiza la tabla de reseñas (chunks) usadas como contexto."""
    if contexto_metadata is None or len(contexto_metadata) == 0:
        return [html.P("Aún no se ha realizado ninguna búsqueda.", className='placeholder-metadata')]

    filas = []
    for fila in contexto_metadata:
        nombre_lugar = fila.get('business_name', 'Lugar desconocido')
        chunk = fila.get('chunk', '')
        filas.append(
            html.Div(className='item-contexto', children=[
                html.Span(f"{nombre_lugar}", className='nombre-lugar'),
                html.P(chunk, className='texto-chunk'),
            ])
        )
    return filas


def render_categorias(pregunta_metadata):
    """Renderiza las categorías (label/score) detectadas para la pregunta."""
    if not pregunta_metadata:
        return [html.P("Aún no se ha clasificado ninguna pregunta.", className='placeholder-metadata')]

    filas = []
    for item in pregunta_metadata:
        label = item.get('label', '—')
        score = float(item.get('score', 0))
        score_pct = round(score * 100, 1)
        filas.append(
            html.Div(className='item-categoria', children=[
                html.Div(className='fila-categoria-superior', children=[
                    html.Span(label, className='etiqueta-categoria'),
                    html.Span(f'{score_pct}%', className='valor-categoria'),
                ]),
                html.Div(className='barra-fondo', children=[
                    html.Div(className='barra-relleno', style={'width': f'{score_pct}%'})
                ]),
            ])
        )
    return filas


# ====================================================================
# 3. APP DASH
# ====================================================================
app = dash.Dash(__name__)
app.title = "Chatbot de Reseñas Turísticas"

app.layout = html.Div(id='fondo-principal', children=[
    html.Div(className='overlay-oscuro', children=[

        # ---------------- Encabezado ----------------
        html.Div(className='header', children=[
            html.H1("Chatbot de Reseñas Turísticas", className='titulo'),
            html.P(
                "Pregunta sobre atracciones, hoteles o restaurantes basándote en reseñas reales",
                className='subtitulo'
            ),
        ]),

        # ---------------- Panel de metadatos ----------------
        html.Div(className='panel-metadata', children=[
            html.Div(className='metadata-columna', children=[
                html.H3("Reseñas utilizadas como contexto"),
                html.Div(id='contenido-contexto', className='caja-scroll',
                         children=render_contexto(None)),
            ]),
            html.Div(className='metadata-columna', children=[
                html.H3("Clasificación de la pregunta"),
                html.Div(id='contenido-categorias', className='caja-scroll',
                         children=render_categorias(None)),
            ]),
        ]),

        # ---------------- Selector de modelo ----------------
        html.Div(className='panel-config', children=[
            html.Div(className='config-item', children=[
                html.Label("Modelo de IA:", className='label-config'),
                dcc.Dropdown(
                    id='selector-modelo',
                    options=OPCIONES_MODELO,
                    value=MODELOS_LOCALES[0],
                    clearable=False,
                    className='dropdown-modelo',
                ),
            ]),
            html.Div(id='contenedor-api-key', className='config-item oculto', children=[
                html.Label("Gemini API Key:", className='label-config'),
                dcc.Input(
                    id='input-api-key',
                    type='password',
                    placeholder='Ingresa tu API key de Gemini',
                    className='input-api-key',
                ),
            ]),
        ]),

        # ---------------- Ventana de chat ----------------
        html.Div(className='ventana-chat', children=[
            dcc.Loading(
                id='loading-chat',
                type='circle',
                color='#ffffff',
                children=html.Div(
                    id='contenedor-mensajes',
                    className='contenedor-mensajes',
                    children=[
                        burbuja_mensaje(
                            'bot',
                            '¡Hola! Pregúntame algo sobre las reseñas turísticas, '
                            'por ejemplo: "¿Las atracciones están limpias?"'
                        )
                    ],
                ),
            )
        ]),

        # ---------------- Entrada de texto ----------------
        html.Div(className='panel-entrada', children=[
            dcc.Input(
                id='input-pregunta',
                type='text',
                placeholder='Escribe tu pregunta sobre las reseñas...',
                className='input-pregunta',
                debounce=False,
                autoComplete='off',
            ),
            html.Button('Enviar ➤', id='boton-enviar', className='boton-enviar', n_clicks=0),
        ]),

        # ---------------- Stores (estado persistente en el navegador) ----------------
        dcc.Store(id='store-historial', data=[]),
        dcc.Store(id='store-mensajes-render', data=[]),
    ])
])


# ====================================================================
# 4. CALLBACKS
# ====================================================================

@app.callback(
    Output('contenedor-api-key', 'className'),
    Input('selector-modelo', 'value'),
)
def alternar_visibilidad_api_key(modelo):
    """Muestra el campo de API key solo cuando se elige el modelo de Gemini."""
    if modelo == MODELO_GEMINI:
        return 'config-item'
    return 'config-item oculto'


@app.callback(
    Output('store-historial', 'data'),
    Output('store-mensajes-render', 'data'),
    Output('contenedor-mensajes', 'children'),
    Output('contenido-contexto', 'children'),
    Output('contenido-categorias', 'children'),
    Output('input-pregunta', 'value'),
    Input('boton-enviar', 'n_clicks'),
    Input('input-pregunta', 'n_submit'),
    State('input-pregunta', 'value'),
    State('selector-modelo', 'value'),
    State('input-api-key', 'value'),
    State('store-historial', 'data'),
    State('store-mensajes-render', 'data'),
    prevent_initial_call=True,
)
def procesar_pregunta(n_clicks, n_submit, pregunta, modelo, api_key,
                       historial_chat, mensajes_render):
    if not pregunta or not pregunta.strip():
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update)

    historial_chat = historial_chat or []
    mensajes_render = mensajes_render or []
    pregunta = pregunta.strip()

    try:
        # ---- 1. Búsqueda semántica (RAG) ----
        resultados_faiss = buscar_en_estrategia(
            pregunta=pregunta,
            nombre_estrategia='oraciones',
            modelo=modelo_e5,
            top_k=5,
        )

        # ---- 2. Filtrado / clasificación ----
        contexto_metadata, pregunta_metadata = filtrador.procesar(
            pregunta=pregunta,
            resultados_faiss=resultados_faiss,
        )

        contexto_csv = contexto_metadata[['business_name', 'chunk']].to_csv(
            index=False,
            header=['lugar', 'reseña'],
        )

        # ---- 3. Generación de la respuesta ----
        if modelo == MODELO_GEMINI:
            respuesta, historial_chat = responder_ia(
                pregunta=pregunta,
                contexto=contexto_csv,
                historial=historial_chat,
                modelo=modelo,
                gemini_api_key=api_key or '',
            )
        else:
            respuesta, historial_chat = responder_ia(
                pregunta=pregunta,
                contexto=contexto_csv,
                historial=historial_chat,
                modelo=modelo,
            )

        # ---- 4. Actualizar historial "de display" (solo para las burbujas) ----
        mensajes_render.append({'rol': 'user', 'texto': pregunta})
        mensajes_render.append({'rol': 'bot', 'texto': respuesta})

        mensajes = [
            burbuja_mensaje(m['rol'], m['texto'])
            for m in mensajes_render
        ]

        # ---- 5. Metadatos para los paneles superiores ----
        contexto_records = contexto_metadata[['business_name', 'chunk']].to_dict('records')

        return (
            historial_chat,
            mensajes_render,
            mensajes,
            render_contexto(contexto_records),
            render_categorias(pregunta_metadata),
            '',
        )

    except Exception as e:
        mensaje_error = burbuja_mensaje('bot', f"Ocurrió un error al procesar la pregunta: {e}")
        mensajes_previos = [
            burbuja_mensaje(m['rol'], m['texto'])
            for m in mensajes_render
        ]
        return (
            historial_chat,
            mensajes_render,
            mensajes_previos + [mensaje_error],
            dash.no_update,
            dash.no_update,
            '',
        )


if __name__ == '__main__':
    app.run(debug=True)