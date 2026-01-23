import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)

# Tüm kaynaklardan gelen isteklere izin ver (CORS)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return "MOTOR ÇALIŞIYOR! (V3.0 - OTOMATİK GÜZELLEŞTİRME AKTİF) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    print("🔔 SİPARİŞ GELDİ! Motor çalışıyor...")
    
    # API Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("❌ HATA: API Token yok!")
        return jsonify({"error": "Sunucu ayarlarında API Token eksik!"}), 500

    # Gelen Veriyi Al
    data = request.json
    if not data:
        data = {}

    # 1. KULLANICININ İSTEĞİ (Örn: "Batman")
    user_prompt = data.get('prompt', 'cyborg')
    url = data.get('url', 'https://google.com')
    strength = float(data.get('strength', 1.45))

    # 2. SİHİRLİ SOS (Otomatik Güzelleştirici) ✨
    # Sen ne yazarsan yaz, arkasına bu kelimeleri ekleyip kaliteyi tavan yaptırıyoruz.
    magic_suffix = ", masterpiece, best quality, highres, 8k, ultra detailed, vibrant, sharp focus, highly detailed, cinematic lighting, distinct image"
    
    # Son Prompt: "Batman" + "Magic Suffix"
    final_prompt = user_prompt + magic_suffix
    
    # 3. KORUMA KALKANI (Negatif Promptlar) 🛡️
    # Bunları asla resme sokma diyoruz.
    neg_prompt = "text, watermark, blur, low quality, ugly, deformed, bad anatomy, disfigured, grainy, broken QR code, distorted, noise, blurry, low resolution"

    print(f"🎨 Çizilen Şey: {final_prompt}")

    try:
        # Replicate Motoruna Gönder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,      # Senin yazdığın + Sihirli Kelimeler
                "negative_prompt": neg_prompt, # Yasaklı kelimeler
                "qr_conditioning_scale": strength,
                "num_inference_steps": 40,
                "guidance_scale": 9.0          # Yapay zekanın hayal gücünü biraz daha özgür bıraktık
            }
        )
        
        # Sonucu Döndür
        resim_linki = str(output[0])
        print("✅ BAŞARILI: Resim üretildi.")
        return jsonify({"image_url": resim_linki})

    except Exception as e:
        print("❌ HATA OLUŞTU:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
