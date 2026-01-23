import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V4.0 Final) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token eksik"}), 500

    try:
        data = request.json or {}
        user_prompt = data.get('prompt', 'teddy bear')
        url = data.get('url', 'https://google.com')
        strength = float(data.get('strength', 1.45))

        # Kalite artirici kelimeler
        final_prompt = user_prompt + ", masterpiece, best quality, 8k, ultra detailed, cinematic lighting, vibrant"
        
        # Yasakli kelimeler
        neg_prompt = "text, watermark, blur, low quality, ugly, deformed, broken QR code"

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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
