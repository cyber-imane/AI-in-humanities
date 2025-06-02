import os
import base64
from typing import Optional
from pathlib import Path
import io
from PIL import Image
import requests
from datetime import datetime

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
                    "text": "Analyze this artwork and describe the artistic style in detail: color palette, brushwork, "
                            "composition, lighting, subject matter, mood, and medium. Use descriptive language suitable "
                            "for AI prompt generation."
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
            raise ValueError(f"No images found in {self.gallery_path}.")

        for img_path in image_paths:
            try:
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
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            raise ValueError("No images were successfully processed.")

    def _retrieve_style_context(self, prompt: str, n_results: int = 2) -> str:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        similar_docs = self.vector_store.similarity_search(prompt, k=n_results)
        combined_style = "\n\n--- STYLE REFERENCE ---\n".join([doc.page_content for doc in similar_docs])
        return combined_style

    def generate_art(self, prompt: str, save_path: Optional[str] = None) -> Optional[str]:
        style_context = self._retrieve_style_context(prompt)

        enhancement_prompt = ChatPromptTemplate.from_template(
            """You are an expert art director.

ARTIST'S STYLE ANALYSIS:
{style_context}

USER'S CREATIVE REQUEST: {user_prompt}

TASK: Create a detailed DALL-E prompt that merges the user's vision with the artist's style.

ENHANCED DALL-E PROMPT:"""
        )

        enhancement_chain = (
            {"style_context": lambda x: style_context, "user_prompt": lambda x: x}
            | enhancement_prompt
            | self.llm
            | StrOutputParser()
        )

        enhanced_prompt = enhancement_chain.invoke(prompt)

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url

            if save_path:
                try:
                    save_dir = Path(save_path)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_prompt = "".join(c if c.isalnum() or c.isspace() else "_" for c in prompt[:30])
                    safe_prompt = "_".join(safe_prompt.split())
                    filename = f"{timestamp}_{safe_prompt}.png"
                    full_path = save_dir / filename

                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        with open(full_path, "wb") as f:
                            f.write(img_response.content)
                except Exception as e:
                    print(f"Error saving image: {e}")

            return image_url
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def generate_random_thematic_prompt(self, theme: str) -> str:
        theme = theme.lower()

        prompt_template = f"""
You are an imaginative prompt engineer. Create a vivid and rich AI art prompt inspired by the theme: '{theme}'. 
Include subjects, scene composition, mood, color palette, and any surreal or symbolic elements. 
Be poetic and evocative — something that inspires a unique DALL·E image.
"""

        response = self.llm.invoke([HumanMessage(content=prompt_template)])
        return response.content.strip()
