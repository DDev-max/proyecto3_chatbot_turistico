import re

def chunking_fijo(texto, tamano_chunk=300, overlap=50):
  chunks = []
  inicio = 0
  while inicio < len(texto):
    fin = inicio + tamano_chunk
    chunk = texto[inicio:fin].strip()
    if chunk:
      chunks.append(chunk)
    inicio = fin - overlap
  return chunks


def chunking_oraciones(texto, oraciones_por_chunk=3, overlap_oraciones=1):
  oraciones = re.split(r'(?<=[.!?])\s+', texto.strip())
  oraciones = [o.strip() for o in oraciones if o.strip()]
  chunks = []
  i = 0
  while i < len(oraciones):
    grupo = oraciones[i : i + oraciones_por_chunk]
    chunk = ' '.join(grupo)
    if chunk:
      chunks.append(chunk)
    i += oraciones_por_chunk - overlap_oraciones
  return chunks


def chunking_parrafos(texto, min_longitud=50):
  parrafos = re.split(r'\n\s*\n', texto)
  parrafos = [p.strip() for p in parrafos if p.strip()]
  chunks = []
  buffer = ''
  for p in parrafos:
    if len(buffer) + len(p) < min_longitud * 3:
      buffer += ' ' + p
    else:
      if buffer.strip():
        chunks.append(buffer.strip())
      buffer = p
  if buffer.strip():
    chunks.append(buffer.strip())
  return chunks