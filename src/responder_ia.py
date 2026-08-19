import ollama
import google.generativeai as genai

def responder_ia(pregunta, contexto, modelo="qwen2.5:7b", gemini_api_key=None):
    prompt_sistema = (
        "Eres un asistente virtual especializado en análisis de reseñas"
        " turísticas.\nResponde a la pregunta del usuario utilizando"
        " ÚNICAMENTE la información proporcionada en el Contexto.\nSi la"
        " respuesta no está en el contexto, di 'No dispongo de suficiente"
        " información en las reseñas para responder'."
    )

    prompt_usuario = (
        f"Contexto relevante:\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
    )

    try:
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            
            modelo_gemini = genai.GenerativeModel(
                model_name=modelo,
                system_instruction=prompt_sistema
            )
            
            response = modelo_gemini.generate_content(
                prompt_usuario,
                generation_config=genai.GenerationConfig(temperature=0.2)
            )
            
            return response.text

        else:
            response = ollama.chat(
                model=modelo,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                options={
                    "temperature": 0.2
                },
            )
            
            return response["message"]["content"]

    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}"