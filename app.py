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
    return "MOTOR HAZIR (V11.0 - Prompt Mühendisliği) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token yok"}), 500

    try:
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'mekanik robot')
        url = data.get('url', 'https://google.com')
        
        # Ayarları panelden alıyoruz
        strength = float(data.get('strength', 1.65))
        guidance = float(data.get('guidance_scale', 9.0))

        # ÇEVİRİ
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
        except Exception as e:
            print(f"Ceviri hatasi: {e}")

        # --- BURASI DEĞİŞTİ: POZİTİF PROMPT MÜHENDİSLİĞİ ---
        # Resim ne olursa olsun, onu "kareli" ve "teknolojik" yapmaya zorluyoruz.
        # "perfect qr code texture" ve "distinct modules" kelimeleri noktaları ayırır.
        style_suffix = ", qr code style, geometric, distinct square modules, perfect alignment, high contrast, 3d render, octane render, vibrant neon colors, 8k, highly detailed, sharp focus"
        
        final_prompt = f"{core_prompt}{style_suffix}"
        
        # --- BURASI DEĞİŞTİ: NEGATİF PROMPT MÜHENDİSLİĞİ ---
        # "fused modules" (erimiş modüller) ve "organic covering qr" (qr'ı kapatan organik şekiller) yasaklandı.
        # Bu sayede papağanın kanadı QR noktasının üstüne binemez.
        neg_prompt = "ugly, blurry, low quality, nsfw, text, watermark, fused modules, melting qr code, organic shapes covering qr, distorted patterns, broken alignment, unreadable code, blurry modules, feather texture covering code, messy"

        print(f"Final Prompt: {final_prompt}")

        # MOTORA GÖNDER
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 50,
                "guidance_scale": guidance,
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
