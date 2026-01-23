import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)

# --- KRİTİK AYAR: TÜM GİRİŞLERE İZİN VER (CORS) ---
# Hem telefondan, hem GitHub'dan, hem PC'den gelen isteği kabul et:
CORS(app, resources={r"/*": {"origins": "*"}})

# API TOKEN (Render'dan çekecek)
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")

@app.route('/')
def home():
    return "MOTOR ÇALIŞIYOR! (V2.0 - CORS AÇIK) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # Token kontrolü
    if not replicate_api_token:
        print("HATA: API Token bulunamadi!")
        return jsonify({"error": "Sunucu ayarlarinda API Token eksik!"}), 500

    data = request.json
    # Veri gelmezse varsayılanları kullan
    if not data:
        data = {}
        
    user_prompt = data.get('prompt', 'A futuristic city')
    url = data.get('url', 'https://google.com')
    
    # Otomatik Kalite Eklentisi
    magic_suffix = ", masterpiece, best quality, highres, 8k, clean qr code, scannable, high contrast"
    final_prompt = user_prompt + magic_suffix
    
    print(f"🎨 İSTEK GELDİ: {final_prompt}")

    try:
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "qr_conditioning_scale": 1.45,
                "num_inference_steps": 40,
                "guidance_scale": 8.0,
                "negative_prompt": "blurry, low quality, ugly, disfigured, text, watermark, deformed"
            }
        )
        # Çıktıyı string'e çevir
        resim_linki = str(output[0])
        print("✅ BAŞARILI:", resim_linki)
        return jsonify({"image_url": resim_linki})
    
    except Exception as e:
        print("❌ HATA:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)