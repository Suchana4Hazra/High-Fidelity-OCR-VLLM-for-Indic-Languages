"""
Simple CLI demo for Indic OCR VLLM
"""
import argparse
from pathlib import Path
import sys
import json

# Add project root (so `src` package is importable)
sys.path.append(str(Path(__file__).parent))

from src.models.ocr_model import PaddleOCRVL
from src.data.synthetic_generator import SyntheticDataGenerator
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(log_level="INFO", log_file="logs/demo.log")


def demo_ocr(image_path: str, output_path: str = None, output_format: str = "json"):
    """
    Demonstrate OCR on a single image
    
    Args:
        image_path: Path to input image
        output_path: Path to save results (optional)
        output_format: Output format (json, markdown, html)
    """
    logger.info("=" * 60)
    logger.info("Indic OCR VLLM Demo")
    logger.info("=" * 60)
    
    # Initialize OCR model
    logger.info("Initializing OCR model...")
    ocr_model = PaddleOCRVL(use_gpu=False, num_threads=4)
    
    # Process image
    logger.info(f"Processing image: {image_path}")
    results = ocr_model.process_image(image_path, output_format=output_format)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("OCR RESULTS")
    logger.info("=" * 60)
    logger.info(f"Image: {results['image_path']}")
    logger.info(f"Regions Detected: {results['num_regions']}")
    logger.info("\nFull Text:")
    logger.info("-" * 60)
    print(results['full_text'])
    logger.info("-" * 60)
    
    logger.info("\nDetailed Regions:")
    for idx, region in enumerate(results['regions'], 1):
        logger.info(f"\nRegion {idx}:")
        logger.info(f"  Text: {region['text']}")
        logger.info(f"  Confidence: {region['confidence']:.2f}")
        logger.info(f"  BBox: {region['bbox']}")
    
    # Save results if output path provided
    if output_path:
        ocr_model.save_results(results, output_path, format=output_format)
        logger.info(f"\nResults saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


def demo_synthetic_data(num_samples: int = 10, output_dir: str = "data/synthetic"):
    """
    Demonstrate synthetic data generation
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("Synthetic Data Generation Demo")
    logger.info("=" * 60)
    
    # Initialize generator
    logger.info("Initializing synthetic data generator...")
    generator = SyntheticDataGenerator(
        languages=["hi", "en"],
        domains=["legal", "medical", "general"]
    )
    
    # Generate data
    logger.info(f"Generating {num_samples} synthetic samples...")
    generator.generate_dataset(
        num_samples=num_samples,
        output_dir=output_dir,
        split="demo"
    )
    
    logger.info("=" * 60)
    logger.info("Synthetic data generation completed!")
    logger.info(f"Check output at: {output_dir}/demo/")
    logger.info("=" * 60)


def demo_batch_ocr(input_dir: str, output_dir: str):
    """
    Demonstrate batch OCR processing
    
    Args:
        input_dir: Directory containing images
        output_dir: Output directory for results
    """
    logger.info("=" * 60)
    logger.info("Batch OCR Processing Demo")
    logger.info("=" * 60)
    
    # Initialize OCR model
    ocr_model = PaddleOCRVL(use_gpu=False, num_threads=4)
    
    # Get all images
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, image_file in enumerate(image_files, 1):
        logger.info(f"\nProcessing {idx}/{len(image_files)}: {image_file.name}")
        
        try:
            results = ocr_model.process_image(str(image_file))
            
            # Save results
            output_file = output_path / f"{image_file.stem}_ocr.json"
            ocr_model.save_results(results, str(output_file), format="json")
            
            logger.info(f"  Detected {results['num_regions']} regions")
            logger.info(f"  Saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Batch processing completed!")
    logger.info("=" * 60)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Indic OCR VLLM Demo Application"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Demo commands")
    
    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR on an image")
    ocr_parser.add_argument("image", type=str, help="Path to input image")
    ocr_parser.add_argument("--output", type=str, help="Output file path")
    ocr_parser.add_argument("--format", type=str, default="json",
                           choices=["json", "markdown", "html"],
                           help="Output format")
    
    # Synthetic data command
    synth_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    synth_parser.add_argument("--samples", type=int, default=10,
                             help="Number of samples to generate")
    synth_parser.add_argument("--output", type=str, default="data/synthetic",
                             help="Output directory")
    
    # Batch OCR command
    batch_parser = subparsers.add_parser("batch", help="Batch OCR processing")
    batch_parser.add_argument("input_dir", type=str, help="Input directory")
    batch_parser.add_argument("output_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "ocr":
        demo_ocr(args.image, args.output, args.format)
    
    elif args.command == "generate":
        demo_synthetic_data(args.samples, args.output)
    
    elif args.command == "batch":
        demo_batch_ocr(args.input_dir, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
