import pandas as pd
import time
import random
import requests
import re
import math
from bs4 import BeautifulSoup
from .quitar_emojis import quitar_emojis

meses = {
    'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
}

def civitatis_scraper(ciudad, tour, max_pgs):
    """
    Extrae reseñas turisticas deEspaña atraves de Civitatis.
    
    Args:
        ciudad (str): Nombre de la ciudad española
        tour (str): Nombre del tour del cual se desean extraer reseñas
        max_pgs (int): Cantidad maxima de paginas a revisar. Cada pagina equivalen a 20 reseñas

    Returns: 
        pd.DataFrame: Dataframe de todas las reseñas extraidas
    """
    
    nuevas_filas = []

    for pagina in range(1, max_pgs + 1):
        print(f"Extrayendo datos de la página {pagina}...")
        
        url = f"https://www.civitatis.com/es/{ciudad}/{tour}/opiniones/{pagina}/?withText=1"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://www.google.com/" # la pagina anterior. Como se encontro la pagina
        }
        
        try:
            respuesta = requests.get(url, headers=headers, timeout=10)
            respuesta.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(respuesta.text, 'html.parser')

        h1_tag = soup.find("h1")
        lugar = h1_tag.text.split('\n')[0].strip() if h1_tag else "Sin nombre"

        meta_tag = soup.find("meta", {"itemprop": "reviewCount"})
        total_resenias = meta_tag["content"] if meta_tag and meta_tag.has_attr("content") else "0"
        
        comentarios = soup.find_all("div", class_="o-container-opiniones-small")
        
        for comentario in comentarios:
            rating_tag = comentario.find("span", class_="m-rating-stars")
            valoracion = rating_tag.get('title', '') if rating_tag else ''
            estrellas_match = re.findall(r"\d+", valoracion) # encuentra numeros
            estrellas = estrellas_match[0] if estrellas_match else "1"

            txt_tag = comentario.find("div", class_="container-opinion-txt")
            resenia = txt_tag.text.strip() if txt_tag else ""
            resenia = ' '.join(quitar_emojis(resenia).split()) if resenia else ""

            fecha = comentario.find("p", class_="a-opiniones-date").text.strip()
            dia, mes_texto, anio = [parte.strip() for parte in fecha.split('/')]
            mes_num = meses[mes_texto.capitalize()]
            fecha_formateada = f"{mes_num}/{dia.zfill(2)}/{anio}"

            nuevas_filas.append({
                'business_name': lugar,
                'total_reviews': total_resenias,
                'review_rating':  math.ceil(int(estrellas) / 2),
                'review_text': resenia,
                'datetime_utc': fecha_formateada,
                'pais': 'ESP',
                'url_fuente' : url
            })

        time.sleep(random.uniform(6, 25))

    return pd.DataFrame(nuevas_filas)