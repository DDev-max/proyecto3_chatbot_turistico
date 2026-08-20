import google.generativeai as genai
import ollama


def responder_ia(
    pregunta, contexto, historial=None, modelo="qwen2.5:7b", gemini_api_key=None
):
    if historial is None:
        historial = []

    prompt_sistema = (
        "Eres un asistente virtual especializado en análisis de reseñas"
        " turísticas.\nResponde a la pregunta del usuario utilizando"
        " ÚNICAMENTE la información proporcionada en el Contexto.\nSi la"
        " respuesta no está en el contexto, di 'No dispongo de suficiente"
        " información en las reseñas para responder'."
    )

    contenido_usuario = (
        f"Contexto relevante:\n{contexto}\n\nPregunta: {pregunta}"
    )

    try:
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

            chat_history_gemini = []
            for msg in historial:
                role = "model" if msg["role"] == "assistant" else "user"
                chat_history_gemini.append(
                    {"role": role, "parts": [msg["content"]]}
                )

            modelo_gemini = genai.GenerativeModel(
                model_name=modelo, system_instruction=prompt_sistema
            )

            chat = modelo_gemini.start_chat(history=chat_history_gemini)
            response = chat.send_message(
                contenido_usuario,
                generation_config=genai.GenerationConfig(temperature=0.2),
            )
            respuesta_texto = response.text

        else:
            mensajes_ollama = [{"role": "system", "content": prompt_sistema}]
            mensajes_ollama.extend(historial)
            mensajes_ollama.append(
                {"role": "user", "content": contenido_usuario}
            )

            response = ollama.chat(
                model=modelo,
                messages=mensajes_ollama,
                options={"temperature": 0.2},
            )
            respuesta_texto = response["message"]["content"]

        historial.append({"role": "user", "content": contenido_usuario})
        historial.append({"role": "assistant", "content": respuesta_texto})

        return respuesta_texto, historial

    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}", historial