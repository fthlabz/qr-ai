import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator

app = Flask(__name__)
# Tüm izinleri aç
CORS(app)

# Çevirmeni başlat
translator = Translator()

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V6.0 - Okunabilir Neon) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # 1. Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}
        
        # 2. Gelen verileri al
        user_input_tr = data.get('prompt', 'mekanik bir robot')
        url = data.get('url', 'https://google.com')
        
        # --- KRİTİK AYAR DEĞİŞİKLİĞİ ---
        # Önceki ayar (1.35) çok düşüktü, resim QR'ı yutuyordu.
        # 1.55 - 1.65 arası en güvenli bölgedir. Hem şekil belli olur hem resim güzel çıkar.
        strength = float(data.get('strength', 1.60))

        print(f"Kullanıcı Girişi (TR): {user_input_tr}")

        # 3. ÇEVİRİ (TR -> EN)
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
            print(f"Çevrilen Prompt (EN): {core_prompt}")
        except Exception as e:
            print(f"Çeviri hatası: {e}")

        # 4. CORE PROMPT & STİL
        # QR kodun okunması için prompt'a 'clean qr code' ekledim.
        style_suffix = ", 3d render, octane render, vibrant neon colors, volumetric lighting, glowing, hyper realistic, 8k, masterpiece, sharp focus, futuristic, highly detailed"
        
        final_prompt = f"{core_prompt}{style_suffix}"
        
        # 5. NEGATİF PROMPT
        # Bulanıklığı ve bozuk kareleri engellemek için güçlendirildi.
        neg_prompt = "ugly, disfigured, low quality, blurry, nsfw, text, watermark, grainy, distorted, broken QR code, low resolution, monochrome, washed out colors, dull, fading patterns"

        print(f"Motora Giden Final Prompt: {final_prompt}")
        print(f"Ayar - Strength: {strength}")

        # 6. Replicate Motoruna Gönder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength, # 1.60 yaptık (QR şekli belirginleşir)
                "num_inference_steps": 50,
                "guidance_scale": 9.0,    # 12'den 9'a düşürdük (Yapay zeka QR'ı fazla bozmaz)
                "control_guidance_start": 0,
                "control_guidance_end": 1.0
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA OLUSTU: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
