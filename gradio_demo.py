"""
Gradio-based demo application for Indic OCR VLLM
"""
import gradio as gr
import json
from pathlib import Path
import sys

# Add project root (so `src` package is importable)
sys.path.append(str(Path(__file__).parent))

from src.models.ocr_model import PaddleOCRVL
from src.inference.async_pipeline import AsyncOCRPipeline
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(log_level="INFO")

# Initialize OCR model
ocr_model = PaddleOCRVL(use_gpu=False, num_threads=4)


def process_image_demo(image_path, output_format, translate_to=None):
    """
    Process image and return OCR results
    
    Args:
        image_path: Path to input image
        output_format: Output format (json, markdown, html)
        translate_to: Target language for translation
        
    Returns:
        OCR results as formatted text
    """
    try:
        # Process image
        results = ocr_model.process_image(
            image_path,
            output_format=output_format,
            translate_to=translate_to if translate_to != "None" else None
        )
        
        # Format output
        if output_format == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        
        elif output_format == "markdown":
            md_text = f"# OCR Results\n\n"
            md_text += f"**Image**: {results['image_path']}\n\n"
            md_text += f"**Regions Detected**: {results['num_regions']}\n\n"
            md_text += "## Detected Text\n\n"
            
            for region in results['regions']:
                md_text += f"- **Text**: {region['text']}\n"
                md_text += f"  - Confidence: {region['confidence']:.2f}\n"
                md_text += f"  - BBox: {region['bbox']}\n\n"
            
            md_text += "\n## Full Text\n\n"
            md_text += results['full_text']
            
            return md_text
        
        elif output_format == "html":
            html_text = "<html><body>"
            html_text += "<h1>OCR Results</h1>"
            html_text += f"<p><strong>Image:</strong> {results['image_path']}</p>"
            html_text += f"<p><strong>Regions:</strong> {results['num_regions']}</p>"
            html_text += "<h2>Detected Text</h2><ul>"
            
            for region in results['regions']:
                html_text += f"<li>{region['text']} (Confidence: {region['confidence']:.2f})</li>"
            
            html_text += "</ul></body></html>"
            return html_text
        
        else:
            return results['full_text']
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Error: {str(e)}"


def create_demo():
    """Create Gradio demo interface"""
    
    with gr.Blocks(title="Indic OCR VLLM Demo") as demo:
        gr.Markdown("""
        # High-Fidelity OCR VLLM for Indic Languages
        
        This demo showcases the OCR pipeline for processing documents in Indic languages.
        
        ## Features
        - Multi-language support (Hindi, Tamil, Marathi, etc.)
        - Layout detection and reading order prediction
        - Optional translation
        - Multiple output formats
        """)
        
        with gr.Row():
            with gr.Column():
                # Input section
                gr.Markdown("### Input")
                image_input = gr.Image(
                    type="filepath",
                    label="Upload Document Image"
                )
                
                output_format = gr.Radio(
                    choices=["json", "markdown", "html", "text"],
                    value="markdown",
                    label="Output Format"
                )
                
                translate_to = gr.Dropdown(
                    choices=["None", "en", "hi", "ta", "mr"],
                    value="None",
                    label="Translate To (Optional)"
                )
                
                process_btn = gr.Button("Process Document", variant="primary")
            
            with gr.Column():
                # Output section
                gr.Markdown("### Output")
                output_text = gr.Textbox(
                    label="OCR Results",
                    lines=20,
                    max_lines=30
                )
        
        # Example images
        gr.Markdown("### Example Images")
        gr.Examples(
            examples=[
                ["demo/examples/sample1.png", "markdown", "None"],
                ["demo/examples/sample2.png", "json", "en"],
            ],
            inputs=[image_input, output_format, translate_to],
            outputs=output_text,
            fn=process_image_demo,
            cache_examples=False
        )
        
        # Event handlers
        process_btn.click(
            fn=process_image_demo,
            inputs=[image_input, output_format, translate_to],
            outputs=output_text
        )
        
        gr.Markdown("""
        ### About
        
        This OCR system uses:
        - **PaddleOCR-VL-0.9B**: Ultra-compact vision-language model
        - **RT-DETR**: Fast layout detection
        - **Quantization-Aware Training**: For efficient CPU deployment
        
        **Note**: This is a demo version with simplified models.
        Full production version would use the complete PaddleOCR-VL architecture.
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
