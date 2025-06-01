


import os
import base64
from typing import Optional
from pathlib import Path
import io
from PIL import Image
import requests
from datetime import datetime
from dotenv import load_dotenv

#load_dotenv()



from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from openai import OpenAI

class ArtistStyleGenerator:
    def __init__(self, gallery_path: str):
        self.gallery_path = Path(gallery_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.embeddings = OpenAIEmbeddings()
        self.client = OpenAI()
        self.vector_store = None
        
        print("🎨 Initializing Artist Style Generator...")
        print(f"📁 Processing gallery: {self.gallery_path}")
        self._process_gallery()

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)

    def _analyze_image(self, image_path: str) -> str:
        image = self._load_image(image_path)
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
        documents = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [p for p in self.gallery_path.iterdir() if p.suffix.lower() in image_extensions]

        if not image_paths:
            raise ValueError(f"No images found in {self.gallery_path}. Supported formats: {', '.join(image_extensions)}")

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
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Process the gallery first.")

        print(f"🔍 Searching for relevant style patterns for: '{prompt}'")
        similar_docs = self.vector_store.similarity_search(prompt, k=n_results)
        print(f"📚 Found {len(similar_docs)} relevant style references")
        for i, doc in enumerate(similar_docs, 1):
            print(f"   Reference {i}: {doc.metadata['filename']}")
        combined_style = "\n\n--- STYLE REFERENCE ---\n".join([doc.page_content for doc in similar_docs])
        return combined_style

    def generate_art(self, prompt: str, save_path: Optional[str] = None) -> Optional[str]:
        print(f"\n🎨 Generating art for: '{prompt}'")
        style_context = self._retrieve_style_context(prompt)

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

    def generate_random_thematic_prompt(self, theme: str) -> str:
        """
        Generate a fresh random prompt related to the theme using LLM.
        """
        theme = theme.lower()

        prompt_template = f"""
You are an expert prompt creator for AI art generation. Please create a vivid, imaginative, detailed text prompt that
describes an artwork strictly within the theme: '{theme}'. Include descriptions of subjects, settings, mood, color palettes,
and styles that would inspire a rich DALL·E image. Keep it creative, open-ended, and evocative.
"""

        response = self.llm.invoke([HumanMessage(content=prompt_template)])
        return response.content.strip()

    # ... you can keep the interactive_mode and helper methods if you want ...























