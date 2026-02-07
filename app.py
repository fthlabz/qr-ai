import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator

app = Flask(__name__)
# Tüm kaynaklardan gelen isteklere izin ver
CORS(app)
translator = Translator()

@app.route('/')
def home():
    return "MOTOR HAZIR (V12.0 - Final Edition) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # 1. API Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("HATA: Replicate API Token bulunamadı!")
        return jsonify({"error": "Sunucu hatası: API Token eksik."}), 500

    try:
        # 2. Gelen Veriyi Al
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'cyberpunk city')
        url = data.get('url', 'https://google.com')
        
        # Panelden gelen ayarları al (Varsayılanlar güvenli aralıktadır)
        strength = float(data.get('strength', 1.65))
        guidance = float(data.get('guidance_scale', 9.0))

        print(f"📥 İstek Geldi -> Prompt: '{user_input_tr}' | Strength: {strength} | Guidance: {guidance}")

        # 3. ÇEVİRİ (Türkçe -> İngilizce)
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
                print(f"🌍 Çeviri Yapıldı: {core_prompt}")
        except Exception as e:
            print(f"⚠️ Çeviri Hatası (Orijinal kullanılıyor): {e}")

        # 4. PROMPT MÜHENDİSLİĞİ (Sır Burada!) 🎩
        # Kullanıcının fikrini alıp, onu bir "QR Kod Dokusu"na dönüştürüyoruz.
        # Bu yapı, resmin QR kodu bozmasını engeller.
        
        final_prompt = (
            f"A perfectly scannable QR code art of {core_prompt}, "
            "intricate details, distinct square modules, vector style, "
            "high contrast, sharp edges, professional digital art, "
            "vibrant colors, clean composition, 8k resolution, unreal engine 5 render, "
            "geometric patterns, perfect alignment markers."
        )

        # 5. NEGATİF PROMPT (Yasaklılar) 🛡️
        # QR kodun erimesini, bulanıklaşmasını ve bozulmasını engelleyen özel terimler.
        neg_prompt = (
            "ugly, blurry, low quality, nsfw, distorted, "
            "fused modules, melting lines, organic shapes covering markers, "
            "broken qr code, unreadable, chaotic, grainy, "
            "text, watermark, messy lines, rounded corners on markers"
        )

        # 6. REPLICATE MOTORUNA GÖNDER 🚀
        # Model: zylim0702/qr_code_controlnet (Endüstri standardı)
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,  # Senin panelden seçtiğin ayar
                "num_inference_steps": 50,          # Kalite için yüksek adım
                "guidance_scale": guidance,         # Senin panelden seçtiğin ayar
                "control_guidance_start": 0,        # Başlangıçtan itibaren QR'ı koru
                "control_guidance_end": 1.0         # Sonuna kadar QR'ı koru
            }
        )
        
        # 7. Sonucu Döndür
        if output and len(output) > 0:
            print("✅ Görsel Başarıyla Oluşturuldu!")
            return jsonify({"image_url": str(output[0])})
        else:
            return jsonify({"error": "Görsel oluşturulamadı."}), 500

    except Exception as e:
        print(f"❌ KRİTİK HATA: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
