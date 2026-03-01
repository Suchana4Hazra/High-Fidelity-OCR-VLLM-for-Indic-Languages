# OCR Project

A practical OCR pipeline with:
- Web UI via Gradio
- CLI commands for single image, batch, and synthetic data generation
- OCR backends: Hugging Face TrOCR (primary), Tesseract (fallback)

## 1. Prerequisites

- Python 3.10+
- Git
- Tesseract OCR installed on system

### Install Tesseract (Windows)

Using Chocolatey:

```powershell
choco install tesseract -y
```

Verify:

```powershell
tesseract --version
```

## 2. Clone and Setup

```powershell
git clone <REPO_URL>
cd OCR
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Environment Variables (recommended)

Set these in the same terminal session where you run the app:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:HF_OCR_MODEL="microsoft/trocr-base-printed"
# Optional for Hugging Face rate limits/download speed:
# $env:HF_TOKEN="hf_xxx"
```

Notes:
- Use `microsoft/trocr-base-printed` for printed docs/screenshots.
- Use `microsoft/trocr-base-handwritten` for handwriting/signatures.
- First Hugging Face model load downloads weights (~1.3GB), so first run is slow.

## 4. Run Web App

```powershell
python gradio_demo.py
```

Open:
- http://127.0.0.1:7860
- http://localhost:7860

## 5. Run CLI

Show commands:

```powershell
python app.py --help
```

Single image OCR:

```powershell
python app.py ocr .\path\to\image.png --format json --output .\out\result.json
```

Batch OCR:

```powershell
python app.py batch .\input_images .\out
```

Generate synthetic data:

```powershell
python app.py generate --samples 20 --output .\data\synthetic
```

## 6. Troubleshooting

### `tesseract is not installed or it's not in your PATH`

Run:

```powershell
where.exe tesseract
tesseract --version
```

If not found, add path for current session:

```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Then restart app.

### Hugging Face download/probing logs (`404 Not Found`)

Some `404` lines for optional files are normal during model probing. If model eventually loads and app starts, it is fine.

### Duplicate logs

Recent code disables logger propagation; if duplicates still appear, restart the Python process.

## 7. Project Structure

- `gradio_demo.py`: Web UI
- `app.py`: CLI entrypoint
- `src/models/ocr_model.py`: OCR pipeline and backend selection
- `src/data/synthetic_generator.py`: synthetic data generation
- `requirements.txt`: dependencies
