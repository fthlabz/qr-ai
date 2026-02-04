import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "MOTOR CALISIYOR! (V5.0 CORE-PROMPT ENGINE) 🚀"

# 🔒 DEĞİŞMEZ CORE PROMPT
CORE_PROMPT = """
Ultra high-end 8K resolution, cinematic, futuristic, hyper-detailed 3D render.
A fully scannable QR code is the central design element and must remain perfectly readable.
High contrast QR modules, sharp edges, no distortion, no perspective warping.
The QR code is naturally integrated into the scene as part of the design, not overlaid.
Professional lighting, global illumination, depth, premium composition.
Cybernetic, modern, print-ready, gallery-quality visual.
"""

# 🧠 SENARYO HARİTASI
SCENARIOS = {
    "halı": "A luxury 30x40 cm carpet where the QR code is woven into the textile pattern, premium fabric texture",
    "lavanta": "A cinematic lavender field at golden hour, the QR code formed naturally by lavender rows",
    "lavanta bahçesi": "A cinematic lavender field at golden hour, the QR code formed naturally by lavender rows",
    "ankastre": "A modern built-in kitchen with oven, dishwasher, refrigerator, the QR code integrated into the kitchen design",
    "buzdolabı": "A modern kitchen refrigerator, the QR code integrated into the door design",
    "çamaşır makinası": "A modern washing machine with the QR code integrated into the front panel",
    "bulaşık makinası": "A modern dishwasher with the QR code integrated into the front panel"
}

# 🛡️ GÜÇLÜ NEGATIVE PROMPT
NEGATIVE_PROMPT = """
unreadable qr code, broken qr modules, missing qr patterns,
distorted squares, perspective warped qr,
low contrast qr, noisy qr, blurred qr,
extra symbols, random text, letters, numbers,
watermark, logo, cropped qr,
tilted qr, partial qr, cut off edges,
low resolution, jpeg artifacts, oversharpen
"""

@app.route('/generate-qr', methods=['POST'])
def generate_qr():
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return jsonify({"error": "API Token bulunamadi"}), 500

    try:
        data = request.json or {}

        user_prompt = data.get('prompt', '').strip().lower()
        url = data.get('url', 'https://google.com')
        strength = float(data.get('strength', 1.45))

        # 🎯 AKILLI PROMPT OLUŞTURMA
        if user_prompt in SCENARIOS:
            scene_prompt = SCENARIOS[user_prompt]
        else:
            scene_prompt = user_prompt

        final_prompt = f"""
        {CORE_PROMPT}
        Scene description: {scene_prompt}
        masterpiece, best quality, ultra detailed, cinematic lighting, sharp focus
        """

        print("FINAL PROMPT:", final_prompt)

        output = replicate.run(
            "zylim0702/qr_code_controlnet:628e604e13cf63d8ec58bd4d238474e8986b054bc5e1326e50995fdbc851c557",
            input={
                "url": url,
                "prompt": final_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "qr_conditioning_scale": min(max(strength, 0.8), 1.8),
                "num_inference_steps": 40,
                "guidance_scale": 8.5
            }
        )

        return jsonify({"image_url": str(output[0])})

    except Exception as e:
        print("HATA:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
