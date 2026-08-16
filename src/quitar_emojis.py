import emoji

def quitar_emojis(texto):
    if isinstance(texto, str):
        return emoji.replace_emoji(texto, replace='')
    return texto