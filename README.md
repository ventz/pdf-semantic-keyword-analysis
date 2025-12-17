# PDF Semantic Keyword Analysis Tool

High-performance, scalable PDF semantic keyword analysis tool using OpenAI's latest reasoning models for intelligent text extraction and thematic matching. Designed to efficiently process thousands to hundreds of thousands of PDFs with smart caching, batch processing, and resume capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Performance & Scalability](#performance--scalability)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Advanced Usage](#advanced-usage)
- [License](#license)

## Overview

This tool performs semantic analysis of PDF documents to identify thematic content related to your keywords. Unlike simple keyword matching, it uses OpenAI's GPT-5.2 reasoning model with high-effort reasoning to:

1. Expand each keyword into 15-20 related semantic topics and concepts
2. Analyze documents for thematic relevance using these expanded topics
3. Extract contextual passages with page numbers for high-confidence matches

This makes it ideal for compliance scanning, document analysis, and content discovery where you need to find documents discussing themes, not just exact word matches.

## Features

- **Semantic Analysis with AI Reasoning**:
  - Uses OpenAI GPT-5.2 with high reasoning effort for deep thematic understanding
  - Automatic keyword expansion into semantic topic clusters (15-20 topics per keyword)
  - Three-phase analysis pipeline for cost efficiency
  - High-confidence matching with subtopic tracking

- **Multi-Library PDF Support**: Automatically tries pikepdf first (fastest), then falls back to PyMuPDF, then pypdf for maximum compatibility

- **Command-Line Interface**: Comprehensive CLI with argparse for all configuration options

- **High Performance**:
  - Concurrent PDF processing with configurable thread pools
  - Parallel API calls with rate limiting
  - Batch processing for memory efficiency
  - Smart multi-level caching (keyword expansion + analysis results)
  - Three-phase filtering pipeline minimizes expensive API calls

- **Scalability**: Designed to handle hundreds of thousands of PDFs
  - Automatic garbage collection
  - Memory-efficient streaming
  - Batch processing with configurable sizes

- **Resilient**:
  - Automatic retry with exponential backoff
  - Resume capability (interrupted processing can be resumed)
  - Comprehensive error logging

- **Flexible Configuration**: All parameters configurable via command-line arguments or environment variables

- **Detailed Output**: JSON output with page numbers, matched subtopics, and text context for each thematic match

## Installation

### Using `uv` (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or download this repository
cd pdf-semantic-keyword-analysis

# Install dependencies
uv sync

# Set up environment variables
cp .env.sample .env
# Edit .env and add your OpenAI API key
```

### Using `pip`

```bash
# Install dependencies
pip install pikepdf pymupdf pypdf openai tqdm

# Set up environment variables
cp .env.sample .env
# Edit .env and add your OpenAI API key
```

## Quick Start

1. **Create a keywords file** (`keywords.txt`):
```text
agnostic
vendor
passwd
```

2. **Add your PDFs** to the `files/` directory

3. **Run the analysis**:
```bash
uv run python app.py
```

4. **Check the results** in `output.json`

## Configuration

### Command-Line Options

All configuration options can be set via command-line arguments:

```bash
uv run python app.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pdf-dir` | `files` | Directory containing PDF files to analyze |
| `--keywords-file` | `keywords.txt` | Path to keywords file (one keyword per line) |
| `--output-file` | `output.json` | Path to output JSON file |
| `--error-file` | `errors.log` | Path to error log file |
| `--expansion-model` | `gpt-5.2` | OpenAI model for keyword expansion |
| `--analysis-model` | `gpt-5.2` | OpenAI model for document analysis |
| `--max-workers` | `10` | Maximum concurrent threads for PDF reading |
| `--max-api-concurrency` | `5` | Maximum concurrent API calls |
| `--batch-size` | `50` | Number of PDFs to process per batch |
| `--max-retries` | `3` | Maximum retry attempts for failed API calls |
| `--cache-dir` | `.analysis_cache` | Directory for storing analysis cache |
| `--no-cache` | `false` | Disable caching (re-analyze all files) |
| `--clear-cache` | `false` | Clear cache before processing |
| `--topics-per-keyword` | `18` | Target number of semantic topics per keyword |

### Environment Variables

Set environment variables in `.env` file (copy from `.env.sample`):

```bash
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-...

# Optional: Override default models
EXPANSION_MODEL=gpt-5.2
ANALYSIS_MODEL=gpt-5.2
```

## Usage Examples

### Example 1: Basic Usage

```bash
# Create keywords file
echo -e "security\npassword\nconfidential" > keywords.txt

# Place PDFs in the files directory
mkdir -p files
cp /path/to/your/pdfs/*.pdf files/

# Run analysis
uv run python app.py
```

### Example 2: Custom PDF Directory and Output

```bash
uv run python app.py \
  --pdf-dir /path/to/pdfs \
  --keywords-file my_keywords.txt \
  --output-file results.json
```

### Example 3: Optimize for Performance

```bash
# Use more workers for faster processing
uv run python app.py \
  --max-workers 20 \
  --max-api-concurrency 10 \
  --batch-size 100
```

### Example 4: Clear Cache and Re-analyze

```bash
# Force re-analysis of all PDFs
uv run python app.py --clear-cache

# Or disable caching entirely
uv run python app.py --no-cache
```

### Example 5: Using Different Models

```bash
# Use environment variables
EXPANSION_MODEL=gpt-4o ANALYSIS_MODEL=gpt-4o uv run python app.py

# Or use command-line arguments
uv run python app.py \
  --expansion-model gpt-4o \
  --analysis-model gpt-4o
```

### Example 6: Resume Interrupted Processing

If processing is interrupted (Ctrl+C or crash):

```bash
# Simply run the script again - it will resume from where it left off
uv run python app.py
```

The tool automatically tracks progress and skips already-processed files.

## How It Works

### Semantic Analysis Pipeline

1. **Keyword Expansion** (One-time, cached):
   - Each keyword is expanded into 15-20 semantic topics using GPT-5.2 with high reasoning effort
   - Topics include synonyms, related concepts, technical terms, and contextual phrases
   - Results are cached to `.expanded_keywords.json` for reuse

2. **PDF Discovery and Extraction**:
   - Recursively scans the configured directory for all PDF files
   - Text extraction with multi-library fallback (pikepdf → PyMuPDF → pypdf)
   - Page-by-page extraction with memory management

3. **Three-Phase Analysis** (Cost-optimized):
   - **Phase 1 - String Pre-filter**: Fast regex check for original keywords (no API cost)
   - **Phase 2 - Quick Relevance Check**: Lightweight AI check for thematic relevance (~$0.0001 per document)
   - **Phase 3 - Full Semantic Analysis**: Deep analysis with expanded topics (~$0.001-0.01 per document)

4. **Intelligent Content Selection**:
   - Analyzes first 2 pages plus stratified sample (25%, 50%, 75%, last page)
   - Reduces token usage while maintaining comprehensive coverage

5. **High-Confidence Matching**:
   - Requires multiple related concepts present across paragraphs
   - Tracks which specific subtopics were matched
   - Extracts representative quotes with page numbers

6. **Caching**:
   - Two-level caching: keyword expansion + analysis results
   - Version-tagged cache prevents mixing analysis methods
   - Cache keys based on content hash and original keywords

7. **Batch Processing and Output**:
   - Processes PDFs in configurable batches for memory efficiency
   - Intermediate batch results saved for recovery
   - Final aggregated JSON output with all high-confidence matches

## Performance & Scalability

### Optimizations

- **PDF Library Selection**: Intelligent fallback chain for maximum compatibility
  - pikepdf: Tried first (~10x faster for simple PDFs, but may fail on complex ones)
  - PyMuPDF: Automatic fallback with good balance of speed and compatibility
  - pypdf: Final fallback, most compatible but slower
  - Validates extracted text to detect PDF operators vs. readable content

- **Concurrent Processing**:
  - Thread pool for parallel PDF reading
  - Semaphore-controlled API rate limiting
  - Batch processing prevents memory overflow

- **Caching**: Previously analyzed PDFs are cached (MD5-based cache keys)

- **Memory Management**:
  - Explicit garbage collection
  - Page-by-page processing
  - Batch size limits

### Scaling Recommendations

For processing **thousands of PDFs**:
- Use default settings
- Consider increasing workers: `--max-workers 15-20` (if you have CPU cores available)

For processing **hundreds of thousands of PDFs**:
- Increase batch size: `--batch-size 100-200`
- Consider running multiple instances in parallel with different source directories
- Default model (gpt-5.2) provides best semantic analysis with reasoning
- Monitor disk space for cache directory

### Cost Estimation

Approximate costs using `gpt-5.2`:

**One-time Keyword Expansion** (cached):
- ~$0.002-0.005 per keyword (one-time cost, reused for all PDFs)

**Per-PDF Analysis** (three-phase pipeline):
- Phase 1 (String filter): $0 (no API call)
- Phase 2 (Quick check): ~$0.0001 per document
- Phase 3 (Full semantic): ~$0.001-0.01 per document

**Typical efficiency**:
- Only 10-30% of documents reach Phase 3 (full analysis)
- Estimated cost savings: 70-90% compared to analyzing all documents
- Small PDF (1-10 pages): ~$0.001-0.003 per file
- Medium PDF (10-50 pages): ~$0.003-0.008 per file
- Large PDF (50+ pages): ~$0.008-0.02 per file

**Caching benefits**:
- Keyword expansion cached permanently (until keywords change)
- Analysis results cached per file
- Re-running with same keywords: near-zero cost

## Output Format

### JSON Structure

```json
[
  {
    "filename": "document1.pdf",
    "keyword": "security",
    "keyword_present": true,
    "confidence": "high",
    "matched_subtopics": [
      "authentication",
      "access control",
      "encryption",
      "vulnerability"
    ],
    "data": [
      {
        "actual_page": "5",
        "text": "The security measures implemented include multi-factor authentication and role-based access control...",
        "subtopics_in_passage": ["authentication", "access control"]
      },
      {
        "actual_page": "12",
        "text": "...encryption protocols are required for sensitive data transmission to prevent vulnerabilities...",
        "subtopics_in_passage": ["encryption", "vulnerability"]
      }
    ]
  }
]
```

### Fields Explained

- `filename`: Name of the PDF file
- `keyword`: The original keyword theme being searched
- `keyword_present`: Boolean indicating if theme was found with high confidence
- `confidence`: Confidence level ("high" - only high-confidence matches are returned)
- `matched_subtopics`: List of semantic subtopics that were detected in the document
- `data`: Array of relevant passages (only included for high-confidence matches)
  - `actual_page`: Page number where passage appears
  - `text`: Contextual passage discussing the theme
  - `subtopics_in_passage`: Which specific subtopics appear in this passage

## Troubleshooting

### Common Issues

**Issue**: "No PDF library available" error

**Solution**: Install at least one PDF library:
```bash
uv pip install pikepdf  # Recommended
# or
uv pip install pymupdf
# or
uv pip install pypdf
```

---

**Issue**: OpenAI API rate limiting errors

**Solution**: Reduce `MAX_API_CONCURRENCY` in config:
```python
MAX_API_CONCURRENCY = 2  # Lower value = fewer concurrent requests
```

---

**Issue**: Out of memory errors

**Solution**: Reduce batch size:
```python
BATCH_SIZE = 25  # Lower value = less memory usage
```

---

**Issue**: PDFs not being processed

**Solution**:
1. Check `errors.log` for specific error messages
2. Verify PDFs are not corrupted
3. Try processing a single test PDF first
4. Check that PDF directory path is correct

---

**Issue**: Scanned/Image-based PDFs not extracting text properly

**Problem**: PDFs created by scanning physical documents or with corrupted OCR layers produce garbage text.

**Solution**:
This tool requires PDFs with extractable text. For scanned PDFs:
1. Use OCR preprocessing (e.g., Adobe Acrobat, Tesseract OCR)
2. Convert scanned images to text-based PDFs first
3. The tool automatically falls back through pikepdf → PyMuPDF → pypdf, but none can extract from pure image PDFs

**How to identify**: If Phase 2 filters most PDFs as "not relevant" and `errors.log` is empty, check if your PDFs are scanned documents.

---

**Issue**: Slow processing

**Solution**:
1. Install all PDF libraries for automatic fallback: `uv pip install pikepdf pymupdf pypdf`
2. Increase thread count: `--max-workers 20`
3. Increase API concurrency (if not rate-limited): `--max-api-concurrency 10`
4. Default gpt-5.2 model provides best quality; adjust workers/concurrency for speed

## Limitations

### PDF Format Requirements

**Text-based PDFs Only**: This tool requires PDFs with extractable text. It does **not** perform OCR on scanned documents.

**Supported**:
- ✅ Native digital PDFs (created from Word, LaTeX, web browsers, etc.)
- ✅ PDFs with properly embedded text
- ✅ PDFs with working text layers

**Not Supported**:
- ❌ Scanned documents without OCR
- ❌ Image-only PDFs
- ❌ PDFs with corrupted or incorrectly encoded OCR layers
- ❌ PDFs where text is rendered as graphics/paths

**Workaround**: Pre-process scanned PDFs with OCR tools:
- Adobe Acrobat Pro (commercial)
- Tesseract OCR (open-source)
- OCRmyPDF (command-line tool)

### Model Requirements

- Requires OpenAI API access with GPT-5.2 model availability
- Falls back gracefully if specific models are unavailable

## Advanced Usage

### Semantic Matching Approach

Unlike traditional keyword matching, this tool uses semantic analysis:
- **Keyword Expansion**: Each keyword is expanded into 15-20 related concepts
- **Thematic Understanding**: AI identifies documents discussing the theme, not just exact word matches
- **High-Confidence Filtering**: Only returns matches where multiple related concepts appear across paragraphs
- **Subtopic Tracking**: Shows which specific semantic concepts were detected

Example: Searching for "vendor" will also find documents discussing:
- supplier, provider, contractor
- third-party, outsourcing
- procurement, sourcing
- vendor management, vendor risk, etc.

### Clearing Cache

Clear all caches and force re-analysis:

```bash
# Using command-line flag (recommended)
uv run python app.py --clear-cache

# Or manually
rm -rf .analysis_cache
rm .expanded_keywords.json
rm .progress.json
```

Disable caching for a single run:

```bash
uv run python app.py --no-cache
```

### Batch Results

Intermediate batch results are saved as `batch_N_results.json`. These can be useful for:
- Inspecting progress during long runs
- Recovering partial results if final output fails
- Debugging issues with specific batches

### Error Logging

Check `errors.log` for detailed error information:

```bash
cat errors.log
```

Each error includes:
- Timestamp
- Filename that failed
- Detailed error message

### Integrating with Other Tools

The JSON output can be easily parsed by other tools:

```bash
# Count total keyword matches
jq '[.[] | select(.keyword_present == true)] | length' output.json

# Extract all pages where "security" was found
jq '[.[] | select(.keyword == "security" and .keyword_present == true) | .data[].actual_page] | unique' output.json

# Get list of files containing specific keyword
jq '[.[] | select(.keyword == "password" and .keyword_present == true) | .filename] | unique' output.json
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests (when available)
pytest
```

### Code Formatting

```bash
# Format code
black app.py

# Lint code
ruff check app.py
```

## License

MIT

## Contributing

Pull requests welcome. For major changes, please open an issue first.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
