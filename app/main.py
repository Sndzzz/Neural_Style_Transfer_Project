from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import sys, os, shutil, uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import send_from_directory

# Path ayarları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
STYLE_LIB_DIR = os.path.join(PROJECT_ROOT, 'style_lib')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fast_style_transformer.pth')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
STATIC_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_OUTPUT_FOLDER, exist_ok=True)

# Model / yardımcı importlar
from models.classic_nst import run_classic_nst
from models.fast_nst import apply_style_to_video  # bu fonksiyon video+stil alıp çıktı yolunu döndürmeli

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    template_folder=os.path.join(os.path.dirname(__file__), 'templates')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mod1')
def mod1():
    return render_template('mod1.html')

@app.route('/mod2')
def mod2():
    artists = sorted([d for d in os.listdir(STYLE_LIB_DIR) if os.path.isdir(os.path.join(STYLE_LIB_DIR, d))])
    return render_template('mod2.html', artists=artists)

@app.route('/mod3')
def mod3():
    style_models = [
        {"name": "Candy", "file": "candy.pth", "preview": "candy.jpg"},
        {"name": "Mosaic", "file": "mosaic.pth", "preview": "mosaic.jpg"},
        {"name": "Rain Princess", "file": "rain_princess.pth", "preview": "rain_princess.jpg"},
        {"name": "Udnie", "file": "udnie.pth", "preview": "udnie.jpg"}
    ]
    return render_template('mod3.html', styles=style_models)


@app.route('/mod1/style_transfer', methods=['POST'])
def style_transfer_endpoint():
    try:
        content_file = request.files.get('content_image')
        style_file = request.files.get('style_image')
        if not content_file or not style_file:
            return jsonify({'error': 'Dosyalar eksik'}), 400

        content_path = os.path.join(UPLOAD_FOLDER, 'content_' + secure_filename(content_file.filename))
        style_path = os.path.join(UPLOAD_FOLDER, 'style_' + secure_filename(style_file.filename))
        content_file.save(content_path)
        style_file.save(style_path)

        output_filename = f'mod1_output_{uuid.uuid4()}.jpg'
        temp_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        run_classic_nst(content_path, style_path, temp_output_path)

        if not os.path.exists(temp_output_path):
            return jsonify({'error': 'Çıktı üretilemedi'}), 500

        final_output_path = os.path.join(STATIC_OUTPUT_FOLDER, output_filename)
        shutil.copyfile(temp_output_path, final_output_path)
        return jsonify({'output_path': f'/static/outputs/{output_filename}'})
    except Exception as e:
        print(f"[Mod1 Hata] {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/mod2/style-transfer', methods=['POST'])
def mod2_style_transfer():
    try:
        content = request.files.get('content')
        artist = request.form.get('artist')
        style_name = request.form.get('style')
        if not content or not artist or not style_name:
            return jsonify({'error': 'Eksik parametre'}), 400

        content_path = os.path.join(UPLOAD_FOLDER, secure_filename(content.filename))
        content.save(content_path)

        style_path = os.path.join(STYLE_LIB_DIR, artist, style_name)
        if not os.path.exists(style_path):
            return jsonify({'error': 'Seçilen stil bulunamadı'}), 400

        output_filename = f'mod2_output_{uuid.uuid4()}.jpg'
        temp_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        run_classic_nst(content_path, style_path, temp_output_path)

        if not os.path.exists(temp_output_path):
            return jsonify({'error': 'Çıktı üretilemedi'}), 500

        final_output_path = os.path.join(STATIC_OUTPUT_FOLDER, output_filename)
        shutil.copyfile(temp_output_path, final_output_path)
        return jsonify({'output_image': f'/static/outputs/{output_filename}'})
    except Exception as e:
        print(f"[Mod2 Hata] {e}", file=sys.stderr)
        return jsonify({'error': str(e)}, 500)

@app.route('/mod3/style-transfer', methods=['POST'])
def mod3_style_transfer():
    try:
        video = request.files.get('video')
        model_name = request.form.get('model_name')  # model adı (örneğin: "candy.pth")

        if not video or not model_name:
            return jsonify({'error': 'Eksik parametre'}), 400

        # Video yükle
        video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
        video.save(video_path)

        # Model yolu
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Seçilen model bulunamadı'}), 400
        
        # Çıktı dosyası oluştur
        output_filename = f"mod3_{uuid.uuid4().hex}.mp4"
        temp_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Stil aktarımı
        apply_style_to_video(video_path, temp_output_path, model_path)

        if not os.path.exists(temp_output_path):
            return jsonify({'error': 'Video çıktı üretilemedi'}), 500

        # Statik klasöre kopyala
        final_output_path = os.path.join(STATIC_OUTPUT_FOLDER, output_filename)
        shutil.copyfile(temp_output_path, final_output_path)

        return jsonify({'output_video': f'/static/outputs/{output_filename}'})

    except Exception as e:
        print(f"[Mod3 Hata] {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/get_artists')
def get_artists():
    artists = sorted([d for d in os.listdir(STYLE_LIB_DIR) if os.path.isdir(os.path.join(STYLE_LIB_DIR, d))])
    return jsonify({'artists': artists})

@app.route('/get_styles/<artist>')
def get_styles(artist):
    try:
        artist_dir = os.path.join(STYLE_LIB_DIR, artist)
        if not os.path.isdir(artist_dir):
            return jsonify([]), 200
        styles = [
            f for f in os.listdir(artist_dir)
            if os.path.isfile(os.path.join(artist_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        return jsonify(styles)
    except Exception as e:
        print(f"[get_styles Hata] {e}", file=sys.stderr)
        return jsonify([]), 200

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
