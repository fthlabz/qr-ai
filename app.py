import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator
import qrcode
from PIL import Image

app = Flask(__name__)
CORS(app)
translator = Translator()

@app.route('/')
def home():
    return "MOTOR HAZIR (V22.0 - CLEAN & WHITE MODE) 🚀"

# --- QR ÜRETİCİ ---
def create_high_density_qr(url_data):
    qr = qrcode.QRCode(
        version=10,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=25,
        border=4,
    )
    qr.add_data(url_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token yok"}), 500

    try:
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'cyberpunk city')
        url = data.get('url', 'https://google.com')
        
        # AYARLAR: Varsayılan değerleri düşürdüm (Neon etkisini kırmak için)
        # Guidance 7.5 idealdir. 10 üstü resmi yakar (neon yapar).
        strength = float(data.get('strength', 1.0))  # 1.9 çok yüksek, 1.0-1.1 ideal
        guidance = float(data.get('guidance_scale', 7.5)) # 9.0 -> 7.5'e çektik

        print(f"İstek: '{user_input_tr}' | Str: {strength} | Guid: {guidance}")

        # ÇEVİRİ
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
        except Exception as e:
            print(f"Çeviri hatası: {e}")

        # --- DÜZELTİLEN KISIM BURASI ---
        # "Vibrant colors", "mosaic", "illusion" kelimelerini kaldırdım.
        # Yerine beyaz arka planı zorlayan kelimeler ekledim.
        final_prompt = (
            f"{core_prompt}, "
            "isolated on white background, "  # Beyaz arka planda izole et
            "clean background, "              # Temiz arka plan
            "minimalist, high quality, "      # Minimalist
            "highly detailed, 8k resolution, "
            "soft lighting"                   # Yumuşak ışık (Neon karşıtı)
        )

        # Negatif prompt'a "neon" ve "karmaşa" ekledik
        neg_prompt = (
            "neon, saturated, vibrant, chaotic, " # Neon ve doygun renkleri engelle
            "complex background, patterns, textures, " # Arka plan desenlerini engelle
            "border, frame, margin, ugly, blurry, low quality, "
            "distorted, broken qr code, unreadable"
        )

        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 50,
                "guidance_scale": guidance,
                "control_guidance_start": 0.0,
                "control_guidance_end": 0.8, # Resmi biraz daha serbest bıraktık
                "qr_code_content": url
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
