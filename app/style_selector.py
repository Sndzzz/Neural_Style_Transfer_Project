import os

# Stil kitaplığı kök klasörü
BASE_DIR = os.getcwd()
STYLE_LIB_DIR = os.path.join(BASE_DIR, 'style_lib')

def get_artists():
    """style_lib altındaki sanatçı klasörlerini listeler."""
    if not os.path.exists(STYLE_LIB_DIR):
        return []
    return sorted([
        name for name in os.listdir(STYLE_LIB_DIR)
        if os.path.isdir(os.path.join(STYLE_LIB_DIR, name))
    ])

def get_works_by_artist(artist_name):
    """Verilen sanatçının style_lib altındaki eser dosyalarını listeler."""
    artist_path = os.path.join(STYLE_LIB_DIR, artist_name)
    if not os.path.exists(artist_path):
        return []

    return sorted([
        file for file in os.listdir(artist_path)
        if os.path.isfile(os.path.join(artist_path, file))
           and file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])