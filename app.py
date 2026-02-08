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
    return "MOTOR HAZIR (V13.0 - Çerçevesiz Sanat Modu) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token yok"}), 500

    try:
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'cyberpunk city')
        url = data.get('url', 'https://google.com')
        
        # Panelden gelen ayarlar
        strength = float(data.get('strength', 1.60))
        guidance = float(data.get('guidance_scale', 9.0))

        print(f"İstek: '{user_input_tr}' | Str: {strength} | Guid: {guidance}")

        # ÇEVİRİ
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
        except Exception as e:
            print(f"Çeviri hatası: {e}")

        # --- BURASI DEĞİŞTİ: SANAT PROMPT MÜHENDİSLİĞİ ---
        # Teknik terimler temizlendi. Sanat ve detay odaklı kelimeler geri geldi.
        # "frameless, no borders, edge-to-edge art" komutuyla çerçeve yasaklandı.
        final_prompt = (
            f"A perfectly scannable QR code art of {core_prompt}, "
            "vibrant colors, highly detailed, octane render, unreal engine 5 render, "
            "cinematic lighting, volumetric fog, 8k resolution, masterpiece, "
            "frameless, no borders, edge-to-edge art."
        )

        # --- BURASI DEĞİŞTİ: NEGATİF PROMPT ---
        # "frame", "border", "margin" en başa eklenerek yasaklandı.
        neg_prompt = (
            "frame, border, margin, padding, ornament, ugly, blurry, low quality, nsfw, "
            "distorted, fused modules, melting lines, organic shapes covering markers, "
            "broken qr code, unreadable, chaotic, grainy, text, watermark, "
            "messy lines, rounded corners on markers"
        )

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
                "control_guidance_end": 0.75
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
