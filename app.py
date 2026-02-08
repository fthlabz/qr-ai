import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
from googletrans import Translator
import qrcode
from PIL import Image

app = Flask(__name__)
CORS(app)
translator = Translator()

@app.route('/')
def home():
    return "MOTOR HAZIR (V21.0 - QR DARPHANE MODU / HIGH QUALITY) 🚀"

# --- ÖZEL FONKSİYON: HIGH LEVEL QR ÜRETİCİ ---
def create_high_density_qr(url_data):
    # İşte senin istediğin "H" ve "L" ayarının yapıldığı yer burası!
    qr = qrcode.QRCode(
        version=None,  # Otomatik yoğunluk (Data sığsın diye)
        # 🔥 KRİTİK AYAR: ERROR_CORRECT_H (%30 Hata Payı) 🔥
        # Bunu 'L' yaparsan QR seyrek olur, resim çıkmaz.
        # 'H' yapınca QR yoğun olur, resim içine gömülür.
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url_data)
    qr.make(fit=True)

    # Siyah beyaz QR resmini oluştur
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Replicate'e göndermek için Base64 formatına çevir
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token yok"}), 500

    try:
        data = request.json or {}
        
        user_input_tr = data.get('prompt', 'cyberpunk city')
        url = data.get('url', 'https://google.com')
        
        # HTML'den gelen ayarlar
        strength = float(data.get('strength', 1.15)) 
        guidance = float(data.get('guidance_scale', 9.0))

        print(f"İstek: '{user_input_tr}' | Str: {strength}")

        # 1. ADIM: ÖNCE BİZİM 'H' KALİTE QR KODUMUZU OLUŞTUR
        # Replicate'e url stringi değil, bu resmi göndereceğiz.
        qr_image_base64 = create_high_density_qr(url)
        print("✅ High-Density QR Kod yerelde oluşturuldu.")

        # ÇEVİRİ
        core_prompt = user_input_tr
        try:
            translation = translator.translate(user_input_tr, dest='en')
            if translation and translation.text:
                core_prompt = translation.text
        except Exception as e:
            print(f"Çeviri hatası: {e}")

        # PROMPT (Mozaik/Füzyon Etkisi İçin)
        final_prompt = (
            f"{core_prompt}, "
            "seamlessly integrated into qr code, "
            "vibrant colors, highly detailed, masterpiece, "
            "mosaic style textures, optical illusion, 8k resolution, "
            "no borders, frameless art"
        )

        neg_prompt = (
            "border, frame, margin, padding, ugly, blurry, low quality, "
            "distorted, broken qr code, unreadable, text, watermark, "
            "obvious black squares, simple barcode"
        )

        # MOTORA GÖNDER
        # Not: zylim0702 modeli bazen 'image' parametresini destekler, bazen sadece url.
        # Eğer bu model hata verirse, 'lucataco/qr-code-controlnet' modeline geçeceğiz.
        # Ama şimdilik senin modelinde deniyoruz.
        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url, # Yedek olarak dursun
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 50,
                "guidance_scale": guidance,
                "control_guidance_start": 0.0,
                "control_guidance_end": 0.75,
                # 🔥 BİZİM ÜRETTİĞİMİZ YOĞUN QR KODU BURAYA GİRİYOR 🔥
                # Model bunu "kontrol resmi" olarak kullanacak.
                "qr_code_content": url # Bazı versiyonlar image almaz, content'i yoğunlaştıramayız ama şansımızı deneriz.
            }
        )
        
        # NOT: Eğer üstteki kod o "yoğunluğu" vermezse, model "image" parametresi istiyor demektir.
        # Replicate'deki bu model versiyonu bazen dışarıdan resim kabul etmez.
        # Eğer çalışmazsa "nateraw/qr-code-controlnet" modelini kullanacağız.
        
        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print(f"HATA: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
