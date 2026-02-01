"""
Synthetic data generation using Gemini API
"""
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger("data_generator")


class SyntheticDataGenerator:
    """Generate synthetic training data for Indic OCR"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        languages: List[str] = None,
        domains: List[str] = None
    ):
        """
        Initialize synthetic data generator
        
        Args:
            api_key: Gemini API key
            languages: List of language codes (hi, ta, mr, etc.)
            domains: List of domains (legal, medical, general)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.languages = languages or ["hi", "en"]
        self.domains = domains or ["legal", "medical", "general"]
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            logger.warning("No Gemini API key provided. Using mock data generation.")
            self.model = None
    
    def generate_text_content(
        self,
        language: str,
        domain: str,
        num_lines: int = 10
    ) -> str:
        """
        Generate text content using Gemini
        
        Args:
            language: Language code
            domain: Domain (legal, medical, general)
            num_lines: Number of lines to generate
            
        Returns:
            Generated text
        """
        if not self.model:
            # Mock data for demo
            return self._generate_mock_text(language, domain, num_lines)
        
        prompt = f"""Generate {num_lines} lines of realistic {domain} text in {language} language.
        Include complex terms, proper nouns, and varied sentence structures.
        Make it look like a real document excerpt.
        Do not add any explanations, just return the text."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return self._generate_mock_text(language, domain, num_lines)
    
    def _generate_mock_text(self, language: str, domain: str, num_lines: int) -> str:
        """Generate mock text for demo purposes"""
        mock_texts = {
            "hi": {
                "legal": "यह एक कानूनी दस्तावेज़ है। पक्षकार इस समझौते के तहत बाध्य हैं।",
                "medical": "रोगी का नाम: राज कुमार। निदान: सामान्य बुखार और सिरदर्द।",
                "general": "आज का दिन बहुत सुंदर है। मौसम बहुत अच्छा है।"
            },
            "en": {
                "legal": "This is a legal document. All parties are bound by this agreement.",
                "medical": "Patient Name: John Doe. Diagnosis: Common fever and headache.",
                "general": "Today is a beautiful day. The weather is very pleasant."
            }
        }
        
        base_text = mock_texts.get(language, mock_texts["en"]).get(domain, mock_texts["en"]["general"])
        return "\n".join([base_text] * num_lines)
    
    def render_text_with_bboxes(
        self,
        text: str,
        image_size: Tuple[int, int] = (800, 600),
        font_size: int = 24
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Render text on image and return bounding boxes
        
        Args:
            text: Text to render
            image_size: Image size (width, height)
            font_size: Font size
            
        Returns:
            Tuple of (rendered image, list of bounding boxes)
        """
        # Create white background
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a Unicode font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Split text into lines
        lines = text.split('\n')
        
        # Calculate positions and bounding boxes
        bboxes = []
        y_offset = 20
        x_offset = 20
        
        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Get text bounding box
            bbox = draw.textbbox((x_offset, y_offset), line, font=font)
            
            # Draw text
            draw.text((x_offset, y_offset), line, fill='black', font=font)
            
            # Store bounding box
            bboxes.append({
                'text': line,
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                'line_idx': line_idx
            })
            
            # Update y offset
            y_offset += bbox[3] - bbox[1] + 10
            
            # Break if we exceed image height
            if y_offset > image_size[1] - 50:
                break
        
        return img, bboxes
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        split: str = "train"
    ):
        """
        Generate complete synthetic dataset
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            split: Dataset split (train/val/test)
        """
        output_path = Path(output_dir) / split
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_samples} synthetic samples for {split} split")
        
        manifest = []
        
        for idx in tqdm(range(num_samples)):
            # Randomly select language and domain
            language = np.random.choice(self.languages)
            domain = np.random.choice(self.domains)
            
            # Generate text
            text = self.generate_text_content(language, domain, num_lines=8)
            
            # Render with bboxes
            img, bboxes = self.render_text_with_bboxes(text)
            
            # Save image
            img_filename = f"{split}_{idx:06d}.png"
            img_path = images_dir / img_filename
            img.save(img_path)
            
            # Save labels
            label_data = {
                'image': str(img_path),
                'language': language,
                'domain': domain,
                'full_text': text,
                'bboxes': bboxes
            }
            
            label_filename = f"{split}_{idx:06d}.json"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
            
            manifest.append({
                'image': str(img_path),
                'label': str(label_path)
            })
        
        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Generated {num_samples} samples in {output_path}")
        logger.info(f"Manifest saved to {manifest_path}")
