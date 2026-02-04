import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
# Tüm izinleri aç
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V4.0 Final) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # 1. Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}
        
        # 2. Gelen verileri al
        user_prompt = data.get('prompt', 'teddy bear')
        url = data.get('url', 'https://google.com')
        strength = float(data.get('strength', 1.45))

        # 3. OTOMATIK GUZELLESTIRME (Magic Prompt) ✨
        # Kullanici ne yazarsa yazsin, arkasina bunu ekliyoruz
        magic_suffix = ", masterpiece, best quality, 8k, ultra detailed, cinematic lighting, vibrant colors, sharp focus"
        final_prompt = user_prompt + magic_suffix
        
        # 4. NEGATIF PROMPT (Kotu seyleri engelle) 🛡️
        neg_prompt = "text, watermark, blur, low quality, ugly, deformed, grainy, broken QR code, distorted, low resolution"

        print(f"Islem basladi: {final_prompt}")

        # 5. Motora Gonder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 40,
                "guidance_scale": 9.0
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA OLUSTU: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
