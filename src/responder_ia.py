import google.generativeai as genai
import ollama


def responder_ia(
    pregunta, contexto, historial=None, modelo="qwen2.5:7b", gemini_api_key=None
):
    if historial is None:
        historial = []

    prompt_sistema = """Eres un asistente virtual especializado en reseñas turisticas de viajes unicamente a España y Costa Rica. 
    Tu unico objetivo es sugerir y describir los lugares segun lo que pida el usuario. 
    Tus respuestas deben de basarse unicamente en lo que digan las siguientes reseñas en formato csv. 
    Si la respuesta no se encuentra en las reseñas, debes de decir "No tengo sufiente informacion para contestar esa pregunta". 
    No debes de contestar ningun tipo de preguntas que no esten relacionadas con las reseñas turisticas proporcionadas."""

    contenido_usuario = (
        f"Reseñas:\n{contexto}\n\nPregunta: {pregunta}"
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