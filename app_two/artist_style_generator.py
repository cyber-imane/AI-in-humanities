import os
import base64
from typing import List, Optional
from pathlib import Path
import io
from PIL import Image
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Updated LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough

# For DALL-E integration
from openai import OpenAI

# API key is now loaded from environment variables
# Make sure to set OPENAI_API_KEY in your .env file or environment


class ArtistStyleGenerator:
    def __init__(self, gallery_path: str):
        """
        Initialize the Artist Style Generator.

        Args:
            gallery_path: Path to the folder containing the artist's gallery images
        """
        self.gallery_path = Path(gallery_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.embeddings = OpenAIEmbeddings()
        self.client = OpenAI()
        self.vector_store = None
        
        print("🎨 Initializing Artist Style Generator...")
        print(f"📁 Processing gallery: {self.gallery_path}")
        self._process_gallery()

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from the given path."""
        return Image.open(image_path)

    def _analyze_image(self, image_path: str) -> str:
        """Analyze an image and extract artistic style features."""
        image = self._load_image(image_path)

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this artwork and describe the artistic style in comprehensive detail. "
                            "Focus on: 1) Color palette and color harmony, 2) Brushwork and texture techniques, "
                            "3) Composition and spatial arrangement, 4) Light and shadow usage, "
                            "5) Subject matter and themes, 6) Overall mood and atmosphere, "
                            "7) Technical approach and medium characteristics. "
                            "Provide specific descriptive terms that would help an AI art generator "
                            "recreate this exact artistic style and aesthetic."
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]
        )

        response = self.llm.invoke([message])
        return response.content

    def _process_gallery(self):
        """Process all images in the gallery and create a vector store."""
        documents = []

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [p for p in self.gallery_path.iterdir()
                       if p.suffix.lower() in image_extensions]

        if not image_paths:
            raise ValueError(f"No images found in {self.gallery_path}. "
                           f"Supported formats: {', '.join(image_extensions)}")

        print(f"🖼️  Found {len(image_paths)} images to analyze...")

        for img_path in image_paths:
            try:
                print(f"   Analyzing {img_path.name}...")
                analysis = self._analyze_image(str(img_path))
                
                doc = Document(
                    page_content=analysis,
                    metadata={
                        "image_path": str(img_path), 
                        "filename": img_path.name,
                        "source": "gallery_analysis"
                    }
                )
                documents.append(doc)
                print(f"   ✅ Successfully processed {img_path.name}")
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")

        if documents:
            print(f"🔄 Creating vector store from {len(documents)} analyzed images...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ Vector store created successfully!")
            print(f"🎯 Ready to generate art inspired by the artist's style!\n")
        else:
            raise ValueError("No images were successfully processed. Check the gallery path and image formats.")

    def _retrieve_style_context(self, prompt: str, n_results: int = 2) -> str:
        """Retrieve relevant style information based on the prompt."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Process the gallery first.")

        print(f"🔍 Searching for relevant style patterns for: '{prompt}'")
        similar_docs = self.vector_store.similarity_search(prompt, k=n_results)
        
        print(f"📚 Found {len(similar_docs)} relevant style references")
        for i, doc in enumerate(similar_docs, 1):
            print(f"   Reference {i}: {doc.metadata['filename']}")

        # Combine the most relevant style analyses
        combined_style = "\n\n--- STYLE REFERENCE ---\n".join([doc.page_content for doc in similar_docs])
        return combined_style

    def generate_art(self, prompt: str, save_path: Optional[str] = None) -> str:
        """
        Generate art based on the artist's style and the given prompt.

        Args:
            prompt: Text prompt describing the desired art piece
            save_path: Optional path to save the generated image locally

        Returns:
            URL to the generated image
        """
        print(f"\n🎨 Generating art for: '{prompt}'")
        
        # Retrieve the most relevant style context from the vector store
        style_context = self._retrieve_style_context(prompt)

        # Create a more sophisticated prompt enhancement
        enhancement_prompt = ChatPromptTemplate.from_template(
            """You are an expert art director specialized in recreating specific artistic styles for AI image generation.

            ARTIST'S STYLE ANALYSIS:
            {style_context}

            USER'S CREATIVE REQUEST: {user_prompt}

            TASK: Create a detailed DALL-E prompt that will generate an image matching the user's request while 
            faithfully reproducing the artist's distinctive style. The prompt should be highly specific and include:

            1. The exact subject matter requested by the user
            2. Specific color palette and color relationships from the artist's work
            3. Precise brushwork and texture techniques
            4. Composition and spatial arrangement style
            5. Lighting and atmospheric qualities
            6. Any distinctive artistic techniques or signature elements

            Make the prompt detailed enough that DALL-E will create art that clearly reflects this artist's 
            unique style while depicting the user's requested subject.

            ENHANCED DALL-E PROMPT:"""
        )

        enhancement_chain = (
            {"style_context": lambda x: style_context, "user_prompt": lambda x: x}
            | enhancement_prompt
            | self.llm
            | StrOutputParser()
        )

        print("🔧 Enhancing prompt with artist's style...")
        enhanced_prompt = enhancement_chain.invoke(prompt)
        
        print(f"📝 Enhanced prompt created:")
        print(f"   '{enhanced_prompt[:100]}{'...' if len(enhanced_prompt) > 100 else ''}'")

        print("🖼️  Generating image with DALL-E...")
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            print(f"✅ Image generated successfully!")

            # Save the image locally if requested
            if save_path:
                try:
                    save_dir = Path(save_path)
                    save_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
                    safe_prompt = "_".join(safe_prompt.split())
                    filename = f"{timestamp}_{safe_prompt}.png"
                    full_path = save_dir / filename

                    print(f"💾 Downloading and saving image...")
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        with open(full_path, "wb") as f:
                            f.write(img_response.content)
                        print(f"✅ Image saved to: {full_path}")
                    else:
                        print(f"❌ Failed to download image: HTTP {img_response.status_code}")
                except Exception as e:
                    print(f"❌ Error saving image: {e}")

            return image_url

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None

    def interactive_mode(self, save_path: Optional[str] = "generated_art"):
        """
        Run the generator in interactive mode, allowing users to generate multiple images.
        """
        print("\n" + "="*60)
        print("🎨 ARTIST STYLE GENERATOR - INTERACTIVE MODE")
        print("="*60)
        print("Enter your prompts to generate art inspired by the artist's style.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("Type 'help' for more commands.")
        print("-"*60)

        while True:
            try:
                user_input = input("\n🎯 Enter your art prompt: ").strip()
                
                if not user_input:
                    print("Please enter a prompt or type 'help' for commands.")
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the Artist Style Generator!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'info':
                    self._show_gallery_info()
                    continue
                
                # Generate art
                image_url = self.generate_art(user_input, save_path=save_path)
                
                if image_url:
                    print(f"\n🌟 Your art is ready!")
                    print(f"🔗 View online: {image_url}")
                    if save_path:
                        print(f"📁 Also saved locally in: {save_path}/")
                else:
                    print("❌ Failed to generate image. Please try again.")

            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Artist Style Generator!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Please try again with a different prompt.")

    def _show_help(self):
        """Display help information."""
        print("\n📚 HELP - Available Commands:")
        print("  • Enter any text prompt to generate art")
        print("  • 'help' - Show this help message")
        print("  • 'info' - Show information about the loaded gallery")
        print("  • 'quit', 'exit', or 'q' - Exit the program")
        print("\n💡 Tips for better results:")
        print("  • Be specific about subjects, scenes, or objects")
        print("  • The AI will automatically apply the artist's style")
        print("  • Try different types of subjects to see style variations")

    def _show_gallery_info(self):
        """Display information about the loaded gallery."""
        if self.vector_store:
            num_docs = self.vector_store.index.ntotal
            print(f"\n📊 Gallery Information:")
            print(f"  • Gallery path: {self.gallery_path}")
            print(f"  • Images processed: {num_docs}")
            print(f"  • Vector store: Ready")
            
            # List processed images
            print(f"  • Processed images:")
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            image_paths = [p for p in self.gallery_path.iterdir()
                          if p.suffix.lower() in image_extensions]
            for img_path in image_paths:
                print(f"    - {img_path.name}")
        else:
            print("❌ No gallery information available - vector store not initialized")


# Example usage
def main():
    try:
        gallery_path = "artist_gallery"  # Replace with your actual gallery path
        
        # Initialize the generator
        generator = ArtistStyleGenerator(gallery_path)
        
        # Run in interactive mode
        generator.interactive_mode(save_path="generated_art")
        
    except Exception as e:
        print(f"❌ Error initializing the generator: {e}")
        print("Please check that:")
        print("  1. Your gallery path is correct")
        print("  2. The gallery contains valid image files")
        print("  3. Your OpenAI API key is set correctly")


if __name__ == "__main__":
    main()
