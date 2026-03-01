"""
Simplified OCR model for demo purposes
This is a mock implementation simulating the PaddleOCR-VL architecture
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import json

from ..utils.logger import get_logger

logger = get_logger("ocr_model")


class LayoutDetector:
    """Mock layout detector (simulating RT-DETR)"""
    
    def __init__(self):
        self.confidence_threshold = 0.5
        logger.info("Initialized Layout Detector")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detected regions with bboxes
        """
        # Lightweight text block detection for printed/screenshot content.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Connect characters into line-like components.
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        merged = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = image.shape[:2]
        regions = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter very small/noisy components.
            if w < 30 or h < 12 or (w * h) < 450:
                continue

            pad_x, pad_y = 3, 2
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)
            regions.append({
                'bbox': [x1, y1, x2, y2],
                'type': 'text',
                'confidence': 0.95,
                'reading_order': idx
            })
        
        # Sort by reading order (top to bottom, left to right)
        regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        return regions


class OCRRecognizer:
    """Mock OCR recognizer (simulating PaddleOCR-VL)"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.use_hf = False
        self.use_tesseract = False
        logger.info("Initialized OCR Recognizer")

        # Try to use Hugging Face TrOCR first (best for handwriting)
        hf_model_name = model_path or os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")
        try:
            import torch  # type: ignore
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

            self._torch = torch
            self._processor = TrOCRProcessor.from_pretrained(hf_model_name)
            self._hf_model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hf_model.to(self._device)
            self._hf_model.eval()
            self.use_hf = True
            logger.info(f"Using Hugging Face OCR model: {hf_model_name} on {self._device}")
            return
        except Exception as e:
            logger.warning(f"Hugging Face OCR unavailable, falling back to Tesseract/mock: {e}")
        
        # Try to use pytesseract if available
        try:
            import pytesseract  # type: ignore
            tesseract_cmd = os.getenv("TESSERACT_CMD")
            if tesseract_cmd and Path(tesseract_cmd).exists():
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            self.use_tesseract = True
            logger.info("Using Tesseract OCR for demo")
        except Exception:
            self.use_tesseract = False
            logger.warning("Tesseract not available, using mock OCR")

    def _prepare_for_hf(self, image: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if min(h, w) < 64:
            rgb = cv2.resize(rgb, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        elif min(h, w) < 128:
            rgb = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(rgb)

    def _prepare_for_tesseract(self, image: np.ndarray) -> Image.Image:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        return Image.fromarray(bw)

    @staticmethod
    def _low_quality_text(text: str) -> bool:
        clean = (text or "").strip()
        if len(clean) < 2:
            return True
        alnum = sum(1 for ch in clean if ch.isalnum())
        return alnum == 0
    
    def recognize(self, image_crop: np.ndarray, language: str = "eng") -> Dict:
        """
        Recognize text in image crop
        
        Args:
            image_crop: Cropped image region
            language: Language code for OCR
            
        Returns:
            Recognition result with text and confidence
        """
        if self.use_hf:
            try:
                pil_img = self._prepare_for_hf(image_crop)
                pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values.to(self._device)
                with self._torch.no_grad():
                    generated_ids = self._hf_model.generate(pixel_values, max_new_tokens=96)
                text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if not self._low_quality_text(text):
                    return {
                        'text': text,
                        'confidence': 0.90
                    }
            except Exception as e:
                logger.error(f"Hugging Face OCR error: {e}")

        if self.use_tesseract:
            try:
                import pytesseract
                pil_img = self._prepare_for_tesseract(image_crop)
                text = pytesseract.image_to_string(
                    pil_img,
                    lang=language,
                    config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                )
                confidence = 0.85
                
                return {
                    'text': text.strip(),
                    'confidence': confidence
                }
            except Exception as e:
                logger.error(f"Tesseract error: {e}")
        
        # Mock OCR result
        return {
            'text': "[Detected Text]",
            'confidence': 0.75
        }


class TranslationModule:
    """Mock translation module"""
    
    def __init__(self):
        logger.info("Initialized Translation Module")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text
        
        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Mock translation
        if source_lang == target_lang:
            return text
        
        return f"[Translation from {source_lang} to {target_lang}]: {text}"


class PaddleOCRVL:
    """
    Main OCR pipeline integrating layout detection, recognition, and translation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        num_threads: int = 4
    ):
        """
        Initialize PaddleOCR-VL pipeline
        
        Args:
            model_path: Path to model checkpoint
            use_gpu: Whether to use GPU
            num_threads: Number of threads for CPU inference
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        
        # Initialize components
        self.layout_detector = LayoutDetector()
        self.ocr_recognizer = OCRRecognizer(model_path)
        self.translator = TranslationModule()
        
        logger.info(f"Initialized PaddleOCR-VL (GPU: {use_gpu}, Threads: {num_threads})")
    
    def process_image(
        self,
        image_path: str,
        output_format: str = "json",
        translate_to: Optional[str] = None
    ) -> Dict:
        """
        Process a single image through the OCR pipeline
        
        Args:
            image_path: Path to input image
            output_format: Output format (json, markdown, html)
            translate_to: Target language for translation (optional)
            
        Returns:
            OCR results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Stage 1: Layout Detection
        regions = self.layout_detector.detect(image)
        if not regions:
            h, w = image.shape[:2]
            regions = [{
                'bbox': [0, 0, w, h],
                'type': 'text',
                'confidence': 0.50,
                'reading_order': 0
            }]
        logger.info(f"Detected {len(regions)} text regions")
        
        # Stage 2: OCR Recognition
        results = []
        for region in regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            
            # Crop region
            crop = image[y1:y2, x1:x2]
            
            # Recognize text
            ocr_result = self.ocr_recognizer.recognize(crop)
            
            # Add to results
            result = {
                'bbox': bbox,
                'text': ocr_result['text'],
                'confidence': ocr_result['confidence'],
                'type': region['type']
            }
            
            results.append(result)

        if results and all(not (r["text"] or "").strip() for r in results):
            h, w = image.shape[:2]
            fallback = self.ocr_recognizer.recognize(image)
            results = [{
                'bbox': [0, 0, w, h],
                'text': fallback['text'],
                'confidence': fallback['confidence'],
                'type': 'text'
            }]
        
        # Stage 3: Optional Translation
        if translate_to:
            for result in results:
                result['original_text'] = result['text']
                result['text'] = self.translator.translate(
                    result['text'],
                    source_lang="auto",
                    target_lang=translate_to
                )
        
        # Format output
        output = {
            'image_path': str(image_path),
            'num_regions': len(results),
            'regions': results,
            'full_text': '\n'.join([r['text'] for r in results])
        }
        
        return output
    
    def process_pdf(
        self,
        pdf_path: str,
        output_format: str = "json"
    ) -> List[Dict]:
        """
        Process PDF document
        
        Args:
            pdf_path: Path to PDF file
            output_format: Output format
            
        Returns:
            List of OCR results per page
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            results = []
            for page_idx, img in enumerate(images):
                # Save temporary image
                temp_path = f"/tmp/page_{page_idx}.png"
                img.save(temp_path)
                
                # Process image
                page_result = self.process_image(temp_path, output_format)
                page_result['page_number'] = page_idx + 1
                
                results.append(page_result)
            
            return results
            
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return []
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []
    
    def save_results(self, results: Dict, output_path: str, format: str = "json"):
        """
        Save OCR results to file
        
        Args:
            results: OCR results
            output_path: Output file path
            format: Output format (json, markdown, html)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format == "markdown":
            md_content = f"# OCR Results\n\n"
            md_content += f"**Image**: {results['image_path']}\n\n"
            md_content += f"**Regions Detected**: {results['num_regions']}\n\n"
            md_content += "## Text Content\n\n"
            md_content += results['full_text']
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        elif format == "html":
            html_content = f"""
            <html>
            <head><title>OCR Results</title></head>
            <body>
                <h1>OCR Results</h1>
                <p><strong>Image:</strong> {results['image_path']}</p>
                <p><strong>Regions Detected:</strong> {results['num_regions']}</p>
                <h2>Text Content</h2>
                <pre>{results['full_text']}</pre>
            </body>
            </html>
            """
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        logger.info(f"Results saved to {output_path}")
