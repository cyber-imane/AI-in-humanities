from flask import Flask, request, jsonify, send_from_directory, render_template
from artist_generator import ArtistStyleGenerator  # Assumes this is the filename of your backend class
import os

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize the style generator
generator = ArtistStyleGenerator(gallery_path="gallery")  # Update path as needed

@app.route("/")
def home():
    return render_template("index.html")  # Assumes index.html is in templates/

@app.route("/generate_prompt/<theme>", methods=["GET"])
def generate_prompt(theme):
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
        return jsonify({"error": "Prompt is required."}), 400

    try:
        image_url = generator.generate_art(prompt)
        if image_url:
            return jsonify({"image_url": image_url})
        else:
            return jsonify({"error": "Failed to generate image."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: serve static files if needed
@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(debug=True)

