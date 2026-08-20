import pandas as pd
from transformers import pipeline

class Filtrador_finetuned:
    def __init__(self, ruta_polaridad=r"C:\Users\Yo\Desktop\final-textos\models\modelo_distilbeto_resenias", ruta_categorias=r"C:\Users\Yo\Desktop\final-textos\models\modelo_categorias_distilbeto"):
        self.polarizador = pipeline("text-classification", model=ruta_polaridad, tokenizer=ruta_polaridad)
        self.clasificador = pipeline("text-classification", model=ruta_categorias, tokenizer=ruta_categorias)

    def procesar(self, pregunta, resultados_faiss, ruta_metadata=r"C:\Users\Yo\Desktop\final-textos\data\5k_metadata.csv"):
        pregunta_polaridad = self.polarizador(pregunta)
        pregunta_lugar = self.clasificador(pregunta)

        pred_pol = pregunta_polaridad[0] if isinstance(pregunta_polaridad, list) else pregunta_polaridad
        pred_lug = pregunta_lugar[0] if isinstance(pregunta_lugar, list) else pregunta_lugar

        filtro_polaridad = pred_pol['label'] if pred_pol['score'] >= 0.9 else False
        filtro_lugar = pred_lug['label'] if pred_lug['score'] >= 0.9 else False

        textos = [fila['chunk'] for fila in resultados_faiss]

        categorias = self.clasificador(textos)
        polaridades = self.polarizador(textos)

        clasificaciones = pd.DataFrame([
            {
                'resena_id': fila['resena_id'],
                'cat_nombre': cat['label'],
                'cat_confi': cat['score'],
                'polar_nombre': pol['label'],
                'polar_confi': pol['score'],
                'chunk': fila['chunk']
            }
            for fila, cat, pol in zip(resultados_faiss, categorias, polaridades)
        ])

        dfs_a_unir = []

        if filtro_lugar:
            score_suficiente = clasificaciones[clasificaciones['cat_confi'] > 0.80]
            lugar_filtrado = score_suficiente[score_suficiente['cat_nombre'] == filtro_lugar]
            dfs_a_unir.append(lugar_filtrado)

        if filtro_polaridad:
            score_suficiente = clasificaciones[clasificaciones['polar_confi'] > 0.80]
            polar_filtrado = score_suficiente[score_suficiente['polar_nombre'] == filtro_polaridad]
            dfs_a_unir.append(polar_filtrado)

        if dfs_a_unir:
            resenias_filtradas = pd.concat(dfs_a_unir, axis=0, ignore_index=True)
        else:
            resenias_filtradas = clasificaciones

        resenias_filtradas = resenias_filtradas.drop_duplicates()

        og_df = pd.read_csv(ruta_metadata, sep=';')
        og_df = og_df[['business_name', 'review_rating', 'resena_id', 'categoria']]

        df_metadata = pd.merge(
            og_df, 
            resenias_filtradas, 
            on='resena_id', 
            how='right'
        )

        return df_metadata, [pred_pol, pred_lug]