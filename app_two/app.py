


from flask import Flask, request, jsonify, render_template
from artist_style_generator import ArtistStyleGenerator

app = Flask(__name__)
generator = ArtistStyleGenerator("temp_gallery")  # Adjust your gallery path here

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_prompt")
def generate_prompt():
    theme = request.args.get("theme", "").strip()
    if not theme:
        return jsonify({"error": "Theme parameter missing"}), 400
    try:
        prompt = generator.generate_random_thematic_prompt(theme)
        return jsonify({"prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_art", methods=["POST"])
def generate_art():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt missing"}), 400
    try:
        image_url = generator.generate_art(prompt)
        if image_url:
            return jsonify({"image_url": image_url})
        else:
            return jsonify({"error": "Failed to generate image"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)























