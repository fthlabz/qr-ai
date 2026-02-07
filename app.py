import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator

app = Flask(__name__)
CORS(app)
translator = Translator()

@app.route('/')
def home():
    return "MOTOR HAZIR (V7.0 - Full Kontrol) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token yok"}), 500

    try:
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'mekanik robot')
        url = data.get('url', 'https://google.com')
        
        # --- BURASI DEĞİŞTİ: Ayarları artık senden alıyoruz ---
        # Eğer göndermezsen varsayılan olarak yine iyi değerleri kullanır.
        strength = float(data.get('strength', 1.55))       # QR'a sadakat
        guidance = float(data.get('guidance_scale', 9.0))  # Renk/Prompt gücü

        print(f"Gelen İstek -> Prompt: {user_input_tr} | Str: {strength} | Guid: {guidance}")

        # ÇEVİRİ
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
        except Exception as e:
            print(f"Ceviri hatasi: {e}")

        # STYLE PROMPT (Senin Neon Tarzın)
        style_suffix = ", 3d render, octane render, vibrant neon colors, volumetric lighting, glowing, hyper realistic, 8k, masterpiece, sharp focus, futuristic, highly detailed, masterpiece, best quality, 8k, ultra detailed, cinematic lighting, vibrant colors, sharp focus"
        final_prompt = f"{core_prompt}{style_suffix}"
        
        neg_prompt = "ugly, disfigured, low quality, blurry, nsfw, text, watermark, grainy, distorted, broken QR code, low resolution, monochrome, washed out colors, dull, fading patterns, text, watermark, blur, low quality, ugly, deformed, grainy, broken QR code, distorted, low resolution"

        # MOTORA GÖNDER
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,  # Senin seçtiğin değer
                "num_inference_steps": 50,
                "guidance_scale": guidance,         # Senin seçtiğin değer
                "control_guidance_start": 0,
                "control_guidance_end": 1.0
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
