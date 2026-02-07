import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator

app = Flask(__name__)
# Tüm izinleri aç
CORS(app)

# Çevirmeni başlat (Türkçe -> İngilizce için)
translator = Translator()

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V5.0 - Neon Core Prompt) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # 1. Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}
        
        # 2. Gelen verileri al (Kullanıcı Türkçe girebilir)
        user_input_tr = data.get('prompt', 'mekanik bir robot')
        url = data.get('url', 'https://google.com')
        
        # Bu tarz "Neon/Glow" görsellerde okunabilirlik ve sanat dengesi için en iyi ayar:
        strength = float(data.get('strength', 1.35))

        print(f"Kullanıcı Girişi (TR): {user_input_tr}")

        # 3. OTOMATİK ÇEVİRİ (TR -> EN) 🇹🇷➡️🇺🇸
        # Sen ne yazarsan yaz, arkada İngilizceye çeviriyoruz ki model anlasın.
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
            print(f"Çevrilen Prompt (EN): {core_prompt}")
        except Exception as e:
            print(f"Çeviri servisinde hata (Fallback kullanılıyor): {e}")
            # Çeviri çalışmazsa orijinal girdiyi kullanır

        # 4. CORE PROMPT (SİHİRLİ KISIM) ✨
        # Demir Adam ve Papağan görselindeki o "Parlak, Neon, 3D" havayı veren kodlar burası:
        style_suffix = ", 3d render, octane render, vibrant neon colors, volumetric lighting, glowing, hyper realistic, 8k, masterpiece, sharp focus, futuristic, intricate details, cinematic lighting, black background contrast"
        
        final_prompt = f"{core_prompt}{style_suffix}"
        
        # 5. NEGATİF PROMPT (Bozuklukları Engelle) 🛡️
        neg_prompt = "ugly, disfigured, low quality, blurry, nsfw, text, watermark, grainy, distorted, broken QR code, low resolution, monochrome, washed out colors, dull"

        print(f"Motora Giden Final Prompt: {final_prompt}")

        # 6. Replicate Motoruna Gönder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 50,  # Daha fazla detay için 50 adım
                "guidance_scale": 12.0,     # Renklerin patlaması ve prompta sadakat için yüksek değer
                "control_guidance_start": 0,
                "control_guidance_end": 1.0
            }
        )
        
        # Çıktı URL'sini döndür
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA OLUSTU: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
