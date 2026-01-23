import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

# DİKKAT: Anahtar artık kodun içinde değil!
# Render'ın "Environment Variables" kısmından çekecek.
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")

@app.route('/')
def home():
    return "MOTOR ÇALIŞIYOR (RENDER) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # API Anahtarı yoksa hata ver
    if not replicate_api_token:
        return jsonify({"error": "API Key eksik! Render ayarlarina ekle."}), 500

    data = request.json
    user_prompt = data.get('prompt', '')
    url = data.get('url', 'https://google.com')
    
    # OTOMATİK PİLOT AYARLARI
    magic_suffix = ", masterpiece, best quality, highres, 8k, highly detailed, clean qr code, scannable, high contrast, vivid colors"
    final_prompt = user_prompt + magic_suffix
    
    try:
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "qr_conditioning_scale": 1.45,
                "num_inference_steps": 40,
                "guidance_scale": 8.0,
                "negative_prompt": "blurry, low quality, ugly, disfigured, text, watermark, grain, deformed, bad anatomy"
            }
        )
        return jsonify({"image_url": str(output[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)