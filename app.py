import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V5.0 CORE PROMPT ENGINE) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}

        # Kullanıcı SADECE ne istediğini yazar
        # Örnek: "traditional carpet", "modern sofa", "coffee cup"
        user_object = data.get('prompt', 'product')
        url = data.get('url', 'https://google.com')
        strength = float(data.get('strength', 1.6))

        # ==============================
        # 🔥 CORE PROMPT (GENEL & EVRENSEL)
        # ==============================
        core_prompt = f"""
        {user_object}, premium product photography,
        real object, realistic materials and texture,
        correct proportions, clean professional appearance.

        Studio lighting, soft shadows, high contrast,
        ultra sharp focus, photorealistic, high resolution.

        Minimal composition, product-centered,
        neutral background, showroom / gallery quality.

        The QR code must be REAL, STANDARD and FULLY SCANNABLE,
        perfect square modules, high contrast,
        no decorative elements inside the QR area.
        """

        # ==============================
        # 🛡️ GENEL NEGATIF PROMPT
        # ==============================
        negative_prompt = """
        cartoon, illustration, painting, digital art,
        AI artifacts, fake texture, plastic look,
        distorted proportions, warped shape,
        blurry, out of focus, low resolution,
        noise, grain, oversharpen,
        messy background, clutter,
        people, hands, faces,
        logo, brand name, watermark, text,
        neon colors, unrealistic lighting,
        broken QR code, decorative QR, rounded QR,
        unreadable QR, distorted QR
        """

        final_prompt = core_prompt.strip()
        neg_prompt = negative_prompt.strip()

        print("FINAL PROMPT:", final_prompt)
        print("NEGATIVE PROMPT:", neg_prompt)

        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": neg_prompt,
                "qr_conditioning_scale": strength,
                "num_inference_steps": 40,
                "guidance_scale": 9.0
            }
        )

        return jsonify({
            "status": "ok",
            "image_url": str(output[0])
        })

    except Exception as e:
        print("HATA:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
