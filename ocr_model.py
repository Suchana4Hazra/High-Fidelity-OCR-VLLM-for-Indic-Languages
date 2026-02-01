"""
Simplified OCR model for demo purposes
This is a mock implementation simulating the PaddleOCR-VL architecture
"""
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
        # Simple mock implementation using contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 20 or h < 10:
                continue
            
            regions.append({
                'bbox': [x, y, x + w, y + h],
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
        logger.info("Initialized OCR Recognizer")
        
        # Try to use pytesseract if available
        try:
            import pytesseract
            self.use_tesseract = True
            logger.info("Using Tesseract OCR for demo")
        except:
            self.use_tesseract = False
            logger.warning("Tesseract not available, using mock OCR")
    
    def recognize(self, image_crop: np.ndarray, language: str = "eng+hin") -> Dict:
        """
        Recognize text in image crop
        
        Args:
            image_crop: Cropped image region
            language: Language code for OCR
            
        Returns:
            Recognition result with text and confidence
        """
        if self.use_tesseract:
            try:
                import pytesseract
                
                # Convert to PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
                
                # Perform OCR
                text = pytesseract.image_to_string(pil_img, lang=language)
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
