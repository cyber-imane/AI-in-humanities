import os
import base64
import tempfile
import shutil
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image
from dotenv import load_dotenv

# Import your existing code
from artist_style_generator import ArtistStyleGenerator

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'temp_gallery'
GENERATED_FOLDER = 'generated_art'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global generator instance
generator = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_temp_gallery():
    """Create a temporary gallery directory"""
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return UPLOAD_FOLDER

def ensure_generated_folder():
    """Ensure generated art folder exists"""
    os.makedirs(GENERATED_FOLDER, exist_ok=True)
    return GENERATED_FOLDER

@app.route('/')
def index():
    """Serve the HTML interface"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Error: index.html not found</h1>
        <p>Please make sure the index.html file is in the same directory as app.py</p>
        """, 404

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        print("📁 Received upload request")
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create temp gallery
        gallery_path = create_temp_gallery()
        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(gallery_path, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                print(f"   ✅ Saved: {filename}")
        
        if not uploaded_files:
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        print(f"🎉 Successfully uploaded {len(uploaded_files)} files")
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files,
            'count': len(uploaded_files)
        })
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process-gallery', methods=['POST'])
def process_gallery():
    """Process the uploaded gallery"""
    global generator
    try:
        print("🔄 Starting gallery processing...")
        
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({'error': 'No gallery found. Please upload images first.'}), 400
        
        # Check if there are any images in the folder
        image_files = [f for f in os.listdir(UPLOAD_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
        
        if not image_files:
            return jsonify({'error': 'No valid images found in gallery'}), 400
        
        print(f"📊 Found {len(image_files)} images to process")
        
        # Initialize the generator with the temp gallery
        generator = ArtistStyleGenerator(UPLOAD_FOLDER)
        
        print("✅ Gallery processed successfully!")
        return jsonify({
            'message': 'Gallery processed successfully',
            'status': 'ready',
            'images_processed': len(image_files)
        })
    
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({'error': f'Failed to process gallery: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate_art():
    """Generate art based on prompt"""
    global generator
    try:
        print("🎨 Received art generation request")
        
        if not generator:
            return jsonify({'error': 'Gallery not processed. Please upload and process images first.'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        print(f"🎯 Generating art for prompt: '{prompt}'")
        
        # Ensure generated folder exists
        ensure_generated_folder()
        
        # Generate the art (this calls your existing code)
        image_url = generator.generate_art(prompt, save_path=GENERATED_FOLDER)
        
        if image_url:
            print("✅ Art generated successfully!")
            return jsonify({
                'image_url': image_url,
                'message': 'Art generated successfully',
                'prompt': prompt
            })
        else:
            print("❌ Failed to generate art")
            return jsonify({'error': 'Failed to generate art. Please check your OpenAI API key and try again.'}), 500
    
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return jsonify({'error': f'Failed to generate art: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_key_set = bool(os.getenv('OPENAI_API_KEY'))
    gallery_exists = os.path.exists(UPLOAD_FOLDER)
    
    return jsonify({
        'status': 'healthy',
        'generator_ready': generator is not None,
        'api_key_configured': api_key_set,
        'gallery_folder_exists': gallery_exists,
        'upload_folder': UPLOAD_FOLDER,
        'generated_folder': GENERATED_FOLDER
    })

@app.route('/generated/<filename>')
def serve_generated_image(filename):
    """Serve generated images"""
    try:
        return send_from_directory(GENERATED_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/gallery-info', methods=['GET'])
def gallery_info():
    """Get information about the current gallery"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({'gallery_exists': False, 'images': []})
        
        image_files = [f for f in os.listdir(UPLOAD_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
        
        return jsonify({
            'gallery_exists': True,
            'images': image_files,
            'count': len(image_files),
            'generator_ready': generator is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB per file.'}), 413

if __name__ == '__main__':
    # Check environment
    print("🔧 Starting Artist Style Generator Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🖼️  Generated images folder: {GENERATED_FOLDER}")
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in one of these ways:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-key-here")
        print("2. Export environment variable: export OPENAI_API_KEY=your-key-here")
        exit(1)
    else:
        print(f"✅ OpenAI API key configured (ends with: ...{api_key[-4:]})")
    
    # Check if index.html exists
    if not os.path.exists('index.html'):
        print("⚠️  Warning: index.html not found in current directory")
        print("Please make sure the HTML file is in the same folder as app.py")
    else:
        print("✅ index.html found")
    
    print("\n🚀 Server starting...")
    print("🔗 Local access: http://localhost:5000")
    print("🌐 External access: http://YOUR-VM-IP:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
