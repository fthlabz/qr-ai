import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V6.0 FULL PAGE QR POSTER ENGINE) 🚀"

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}

        # Kullanıcı SADECE ne istediğini yazar
        user_object = data.get('prompt', 'product')
        url = data.get('url', 'https://google.com')
        strength = float(data.get('strength', 1.8))

        # ==============================
        # 🔥 CORE PROMPT (EVRENSEL)
        # ==============================
        core_prompt = f"""
        {user_object}, premium product photography,
        real object, realistic materials and texture,
        correct proportions, clean professional appearance.

        The product is placed in the foreground or beside the QR code,
        NOT overlapping the QR code area.

        The QR code is centered in the background,
        isolated, unobstructed, fully visible.

        The QR code must be REAL, STANDARD and FULLY SCANNABLE,
        perfect square modules, high contrast black and white,
        NO objects, NO decorations, NO artistic elements inside the QR area.

        The final image is a FULL PAGE POSTER DESIGN,
        vertical layout, 30x40 cm print ratio.

        The QR code occupies approximately 20–30 percent of the total page area,
        centered or lower-center aligned.

        Studio lighting, soft shadows,
        ultra sharp focus, photorealistic, high resolution.

        Minimal composition, commercial style,
        neutral background, showroom quality.
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
        objects overlapping QR,
        product inside QR,
        decorative QR, artistic QR,
        rounded QR modules,
        broken QR, unreadable QR,
        empty background, blank page,
        QR only image, isolated QR,
        small canvas, cropped composition
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
                "guidance_scale": 9.0,

                # 🔥 FULL PAGE / POSTER BOYUTU
                "width": 768,
                "height": 1024
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
