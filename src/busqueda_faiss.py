import numpy as np
import faiss
import pandas as pd

def buscar_en_estrategia(
    pregunta, nombre_estrategia, modelo, top_k=2
):
  embeddings = np.load(f'embeddings_{nombre_estrategia}.npy')
  metadata = pd.read_csv(f'metadata_chunks_{nombre_estrategia}.csv')

  dimension = embeddings.shape[1]
  indice = faiss.IndexFlatL2(dimension) 

  faiss.normalize_L2(embeddings)  
  indice.add(embeddings)

  embedding_pregunta = modelo.encode([f'query: {pregunta}'])
  faiss.normalize_L2(embedding_pregunta) 

  distancias, indices = indice.search(embedding_pregunta, top_k)

  resultados = []
  for dist, idx in zip(distancias[0], indices[0]):
    row = metadata.iloc[idx]
    resultados.append({
        'resena_id': row['resena_id'],
        'chunk': row['texto'],
        'score': float(1 - dist),
    })

  return resultados