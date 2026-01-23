import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)

# Tüm kaynaklardan gelen isteklere izin ver (Telefon, PC, GitHub vs.)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return "MOTOR ÇALIŞIYOR! (Ana Kapı Açık) 🚀"

# İŞTE SENİN SORDUĞUN YER: '/generate-qr' BURADA TANITILIYOR 👇
@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    print("🔔 Biri /generate-qr kapısını çaldı!")
    
    # API Anahtarını al
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token (Anahtar) eksik!"}), 500

    data = request.json
    if not data:
        data = {}

    # Gelen verileri al
    prompt = data.get('prompt', 'red samurai')
    url = data.get('url', 'https://google.com')
    strength = float(data.get('strength', 1.45))

    try:
        # Replicate'e gönder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 40,
                "guidance_scale": 8.0
            }
        )
        # Cevabı gönder
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print("HATA:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Bu kısım Render için önemli
    app.run(host='0.0.0.0', port=10000)
