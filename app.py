import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
# Tüm izinleri aç
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V5.0 Core Prompt Edition) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    # 1. Token Kontrolü
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}
        
        # 2. Gelen verileri al
        user_prompt = data.get('prompt', 'teddy bear')
        url = data.get('url', 'https://google.com')
        # Varsayılan gücü 1.85 yaptik ki QR garanti okunsun
        strength = float(data.get('strength', 1.85))

        # ---------------------------------------------------------
        # 3. AKILLI CORE PROMPT SISTEMI (Smart Logic) 🧠
        # ---------------------------------------------------------
        
        # SENARYO 1: Kullanıcı uzun ve detaylı bir İngilizce prompt girmiştir (Pro Mod)
        if len(user_prompt) > 70:
             # Sadece ufak kalite eklemeleri yap, kullanıcının cümlesini bozma
             final_prompt = user_prompt + ", 8k, best quality, masterpiece, sharp focus"
             print(f"MOD: PRO (Kullanici promptu kullanildi)")
             
        # SENARYO 2: Kullanıcı sadece "Hamburger" veya "Samurai" yazmıştır (Otomatik Mod)
        else:
             # MASTER PROMPT DEVREYE GIRER: "Koyu yerleri siyah, açık yerleri beyaz yap" emri verilir.
             final_prompt = f"A high-quality, cinematic, sharp focus image of {user_prompt}. The QR code pattern is seamlessly integrated into the texture of the subject. The DARK ELEMENTS of the subject form the black data points, and the LIGHT ELEMENTS form the white background. No separate stickers, no frames. High contrast, 8k resolution, highly detailed, photorealistic masterpiece, dramatic lighting."
             print(f"MOD: OTO (Master Prompt devreye girdi): {user_prompt}")

        # ---------------------------------------------------------

        # 4. NEGATIF PROMPT (Guclendirilmis Koruma) 🛡️
        # 'Broken QR code' ve 'plastic look' eklendi.
        neg_prompt = "text, watermark, blur, low quality, ugly, deformed, grainy, broken QR code, distorted, low resolution, store shelves, price tags, people, messy background, plastic look, scan artifacts"

        print(f"Islem basladi: {final_prompt[:100]}... | Guc: {strength}")

        # 5. Motora Gonder
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 40,
                "guidance_scale": 7.5 
            }
        )
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA OLUSTU: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
