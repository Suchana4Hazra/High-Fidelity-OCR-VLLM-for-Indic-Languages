"""
Asynchronous inference pipeline for high-throughput OCR
Implements the thread-wise architecture described in the project note
"""
import queue
import threading
from typing import List, Dict, Optional
from pathlib import Path
import time
import numpy as np
import cv2

from ..models.ocr_model import LayoutDetector, OCRRecognizer
from ..utils.logger import get_logger

logger = get_logger("async_pipeline")


class AsyncOCRPipeline:
    """
    Thread-wise asynchronous OCR pipeline
    
    Architecture:
    1. DataLoader Thread (preprocessing)
    2. Layout Analysis Thread (RT-DETR)
    3. VLM OCR Inference Thread (recognition)
    """
    
    def __init__(
        self,
        queue_size: int = 100,
        batch_size: int = 16,
        batch_timeout: float = 0.5,
        num_workers: int = 4
    ):
        """
        Initialize async pipeline
        
        Args:
            queue_size: Maximum queue size
            batch_size: Maximum batch size for OCR
            batch_timeout: Maximum wait time for batch
            num_workers: Number of worker threads
        """
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.num_workers = num_workers
        
        # Initialize queues
        self.queue_a = queue.Queue(maxsize=queue_size)  # Preprocessed images
        self.queue_b = queue.Queue(maxsize=queue_size)  # Detected regions
        self.results_queue = queue.Queue()  # Final results
        
        # Initialize components
        self.layout_detector = LayoutDetector()
        self.ocr_recognizer = OCRRecognizer()
        
        # Thread control
        self.stop_event = threading.Event()
        self.threads = []
        
        logger.info(f"Initialized AsyncOCRPipeline (workers: {num_workers}, batch: {batch_size})")
    
    def start(self):
        """Start all worker threads"""
        logger.info("Starting pipeline threads...")
        
        # Start layout analysis thread
        layout_thread = threading.Thread(
            target=self._layout_analysis_worker,
            name="LayoutAnalysisThread"
        )
        layout_thread.start()
        self.threads.append(layout_thread)
        
        # Start OCR inference threads
        for i in range(self.num_workers):
            ocr_thread = threading.Thread(
                target=self._ocr_inference_worker,
                name=f"OCRInferenceThread-{i}"
            )
            ocr_thread.start()
            self.threads.append(ocr_thread)
        
        logger.info(f"Started {len(self.threads)} worker threads")
    
    def stop(self):
        """Stop all worker threads"""
        logger.info("Stopping pipeline threads...")
        self.stop_event.set()
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        logger.info("Pipeline stopped")
    
    def _preprocess_image(self, image_path: str) -> Dict:
        """
        Preprocess image (DataLoader stage)
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image data
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Basic preprocessing
        # Could add: resizing, normalization, etc.
        
        return {
            'image_path': image_path,
            'image': image,
            'metadata': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }
    
    def _layout_analysis_worker(self):
        """
        Layout analysis worker thread
        Consumes from queue_a, produces to queue_b
        """
        logger.info("Layout analysis worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get preprocessed image from queue_a
                data = self.queue_a.get(timeout=1.0)
                
                # Perform layout detection
                image = data['image']
                regions = self.layout_detector.detect(image)
                
                # Add regions to queue_b
                for region in regions:
                    bbox = region['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Crop region
                    crop = image[y1:y2, x1:x2]
                    
                    region_data = {
                        'image_path': data['image_path'],
                        'crop': crop,
                        'bbox': bbox,
                        'type': region['type'],
                        'reading_order': region['reading_order']
                    }
                    
                    self.queue_b.put(region_data)
                
                self.queue_a.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Layout analysis error: {e}")
        
        logger.info("Layout analysis worker stopped")
    
    def _ocr_inference_worker(self):
        """
        OCR inference worker thread
        Consumes from queue_b, produces to results_queue
        Uses dynamic batching
        """
        logger.info("OCR inference worker started")
        
        batch = []
        last_batch_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Try to get item from queue
                try:
                    region_data = self.queue_b.get(timeout=0.1)
                    batch.append(region_data)
                    self.queue_b.task_done()
                except queue.Empty:
                    pass
                
                # Check if we should process batch
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and time.time() - last_batch_time > self.batch_timeout)
                )
                
                if should_process and batch:
                    # Process batch
                    results = []
                    for item in batch:
                        # Recognize text
                        ocr_result = self.ocr_recognizer.recognize(item['crop'])
                        
                        result = {
                            'image_path': item['image_path'],
                            'bbox': item['bbox'],
                            'text': ocr_result['text'],
                            'confidence': ocr_result['confidence'],
                            'type': item['type'],
                            'reading_order': item['reading_order']
                        }
                        
                        results.append(result)
                    
                    # Add results to queue
                    self.results_queue.put(results)
                    
                    # Reset batch
                    batch = []
                    last_batch_time = time.time()
                
            except Exception as e:
                logger.error(f"OCR inference error: {e}")
        
        logger.info("OCR inference worker stopped")
    
    def process_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images asynchronously
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of OCR results
        """
        # Start pipeline
        self.start()
        
        # Add images to queue_a
        for image_path in image_paths:
            try:
                preprocessed = self._preprocess_image(image_path)
                self.queue_a.put(preprocessed)
            except Exception as e:
                logger.error(f"Error preprocessing {image_path}: {e}")
        
        # Collect results
        all_results = []
        processed_images = set()
        
        # Wait for all results
        while len(processed_images) < len(image_paths):
            try:
                results = self.results_queue.get(timeout=5.0)
                
                # Group results by image
                for result in results:
                    all_results.append(result)
                    processed_images.add(result['image_path'])
                
            except queue.Empty:
                if self.queue_a.empty() and self.queue_b.empty():
                    break
        
        # Stop pipeline
        self.stop()
        
        # Group results by image path
        grouped_results = {}
        for result in all_results:
            img_path = result['image_path']
            if img_path not in grouped_results:
                grouped_results[img_path] = []
            grouped_results[img_path].append(result)
        
        # Sort by reading order and format
        final_results = []
        for img_path, regions in grouped_results.items():
            regions.sort(key=lambda r: r['reading_order'])
            
            final_results.append({
                'image_path': img_path,
                'num_regions': len(regions),
                'regions': regions,
                'full_text': '\n'.join([r['text'] for r in regions])
            })
        
        return final_results
