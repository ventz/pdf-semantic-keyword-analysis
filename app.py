import os
import json
import re
import concurrent.futures
import threading
import argparse
from openai import OpenAI
from tqdm import tqdm
import time
import hashlib
import pickle  # Used only for local cache (not untrusted content)
from datetime import datetime
import gc

# Try to import ALL PDF libraries for fallback support
PDF_LIBRARY = None
AVAILABLE_PDF_LIBS = {}

try:
    import pikepdf
    AVAILABLE_PDF_LIBS['pikepdf'] = True
    if PDF_LIBRARY is None:
        PDF_LIBRARY = "pikepdf"
except ImportError:
    AVAILABLE_PDF_LIBS['pikepdf'] = False

try:
    import fitz  # PyMuPDF
    AVAILABLE_PDF_LIBS['pymupdf'] = True
    if PDF_LIBRARY is None:
        PDF_LIBRARY = "pymupdf"
except ImportError:
    AVAILABLE_PDF_LIBS['pymupdf'] = False

try:
    from pypdf import PdfReader
    AVAILABLE_PDF_LIBS['pypdf'] = True
    if PDF_LIBRARY is None:
        PDF_LIBRARY = "pypdf"
except ImportError:
    AVAILABLE_PDF_LIBS['pypdf'] = False

if PDF_LIBRARY is None:
    raise ImportError(
        "No PDF library available. Install one of: pikepdf (recommended), pymupdf, or pypdf"
    )

print(f"Using {PDF_LIBRARY} (primary) for PDF text extraction")
if sum(AVAILABLE_PDF_LIBS.values()) > 1:
    fallbacks = [k for k, v in AVAILABLE_PDF_LIBS.items() if v and k != PDF_LIBRARY]
    print(f"Fallback libraries available: {', '.join(fallbacks)}")

########################################
# CONFIGURATION
########################################
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PDF Semantic Keyword Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output options
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='files',
        help='Directory containing PDF files to analyze'
    )
    parser.add_argument(
        '--keywords-file',
        type=str,
        default='keywords.txt',
        help='Path to keywords file (one keyword per line)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='output.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--error-file',
        type=str,
        default='errors.log',
        help='Path to error log file'
    )

    # Model configuration
    parser.add_argument(
        '--expansion-model',
        type=str,
        default=os.environ.get("EXPANSION_MODEL", "gpt-5.2"),
        help='OpenAI model for keyword expansion'
    )
    parser.add_argument(
        '--analysis-model',
        type=str,
        default=os.environ.get("ANALYSIS_MODEL", "gpt-5.2"),
        help='OpenAI model for document analysis'
    )

    # Performance options
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Maximum concurrent workers for file I/O (PDF reading)'
    )
    parser.add_argument(
        '--max-api-concurrency',
        type=int,
        default=5,
        help='Maximum concurrent API calls'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing PDFs'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries for API calls'
    )

    # Cache options
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.analysis_cache',
        help='Directory for storing analysis cache'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (re-analyze all files)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before processing'
    )

    # Advanced options
    parser.add_argument(
        '--topics-per-keyword',
        type=int,
        default=18,
        help='Target number of semantic topics per keyword'
    )

    return parser.parse_args()

# Default configuration (will be overridden by command line args)
pdf_directory = 'files'
MAX_WORKERS_THREADS = 10
MAX_API_CONCURRENCY = 5
BATCH_SIZE = 50
CACHE_DIR = '.analysis_cache'
MAX_RETRIES = 3
OUTPUT_FILE = 'output.json'
ERROR_FILE = 'errors.log'
RESUME_FILE = '.progress.json'
KEYWORDS_FILE = "keywords.txt"
EXPANSION_MODEL = os.environ.get("EXPANSION_MODEL", "gpt-5.2")
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "gpt-5.2")
EXPANDED_KEYWORDS_CACHE = '.expanded_keywords.json'
TOPICS_PER_KEYWORD = 18
CACHE_VERSION = "2.0-semantic"
NO_CACHE = False
########################################

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the OpenAI client with your API key
client = OpenAI()

# Semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(MAX_API_CONCURRENCY)

def extract_text_from_response(response):
    """
    Extract text from OpenAI response.output list.
    The output contains ResponseReasoningItem and ResponseOutputMessage objects.
    We need to find the ResponseOutputMessage and extract its text content.
    """
    if not hasattr(response, 'output') or not response.output:
        raise ValueError("Response has no output")

    # response.output is a list that may contain reasoning items and message items
    for item in response.output:
        # Look for the message type (not reasoning)
        if hasattr(item, 'type') and item.type == 'message':
            if hasattr(item, 'content') and item.content:
                # Extract text from the first content item
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        return content_item.text

    # Fallback: if we can't find the message, try converting to string
    return str(response.output)

def load_keywords(keywords_file="keywords.txt"):
    """
    Load keywords from a text file, one keyword per line.
    Falls back to default keywords if file doesn't exist or is empty.
    """
    default_keywords = [
        "agnostic", "vendor", "passwd"
    ]

    try:
        if os.path.exists(keywords_file):
            with open(keywords_file, 'r') as f:
                keywords = [line.strip() for line in f.readlines() if line.strip()]

            if keywords:
                print(f"Loaded {len(keywords)} keywords from {keywords_file}")
                return keywords
            else:
                print(f"Warning: No keywords found in {keywords_file}, using default keywords")
                return default_keywords
        else:
            print(f"Keywords file {keywords_file} not found, using default keywords")
            return default_keywords
    except Exception as e:
        print(f"Error loading keywords from {keywords_file}: {e}. Using default keywords.")
        return default_keywords

def expand_keywords(keywords):
    """
    Expand each keyword into 15-20 related semantic topics.
    Uses OpenAI to generate topic clusters for semantic analysis.
    """
    expanded = {}

    for keyword in keywords:
        prompt = f"""Generate a semantic topic expansion for document analysis.

KEYWORD: "{keyword}"

Task: Generate 18 related concepts that indicate this theme in documents.

Guidelines:
1. Include direct synonyms and variations (3-4)
2. Include specific subtopics and domains (5-6)
3. Include technical terms and jargon (4-5)
4. Include common contextual terms (3-4)

Requirements:
- Single words or 2-3 word phrases only
- Avoid overlap between terms
- Prefer formal/technical language
- No generic terms ("information", "data", "system")
- Terms likely to appear in PDF documents

Output: Return ONLY a JSON array of exactly 18 strings.

Example format:
["term1", "term2", "term3", ...]
"""

        # Retry logic with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = client.responses.create(
                    model=EXPANSION_MODEL,
                    input=prompt,
                    reasoning={"effort": "high"}
                )
                # Extract text from response
                result = extract_text_from_response(response).strip()

                # Clean and parse JSON
                result = clean_json_string(result)
                topics = json.loads(result)

                # Validate response
                if not isinstance(topics, list):
                    raise ValueError("Response is not a JSON array")
                if len(topics) < 10:  # At least 10 topics
                    raise ValueError(f"Only got {len(topics)} topics, expected ~18")

                expanded[keyword] = topics
                print(f"  ✓ {keyword} → {', '.join(topics[:6])}... ({len(topics)} topics)")
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  ⚠ Expansion failed for '{keyword}', retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    # Fall back to using just the original keyword
                    print(f"  ✗ Failed to expand '{keyword}' after {MAX_RETRIES} attempts. Using original keyword only.")
                    expanded[keyword] = [keyword]
                    write_to_error(f"Failed to expand keyword '{keyword}': {e}")

    return expanded

def load_or_expand_keywords(keywords_file):
    """
    Load keywords from file and return both original and expanded versions.
    Uses cached expansion if available and valid.
    """
    # Load original keywords
    original_keywords = load_keywords(keywords_file)

    # Check if expansion cache exists and is valid
    if os.path.exists(EXPANDED_KEYWORDS_CACHE):
        try:
            with open(EXPANDED_KEYWORDS_CACHE, 'r') as f:
                cache = json.load(f)

            # Validate cache contains all current keywords
            cached_keywords = set(cache.get("source_keywords", []))
            current_keywords = set(original_keywords)

            if cached_keywords == current_keywords:
                print("Using cached keyword expansion")
                expanded = cache.get("expanded", {})
                # Display cached expansion
                for keyword, topics in expanded.items():
                    print(f"  ✓ {keyword} → {', '.join(topics[:6])}... ({len(topics)} topics)")
                return original_keywords, expanded
            else:
                print("Keyword list changed, re-expanding...")
        except Exception as e:
            print(f"Cache invalid or corrupted: {e}. Re-expanding keywords...")

    # Need to expand keywords
    print("Expanding keywords semantically...")
    expanded = expand_keywords(original_keywords)

    # Save to cache
    try:
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "source_keywords": original_keywords,
            "expanded": expanded
        }
        with open(EXPANDED_KEYWORDS_CACHE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Expansion cached to {EXPANDED_KEYWORDS_CACHE}")
    except Exception as e:
        print(f"Warning: Could not cache expansion: {e}")

    return original_keywords, expanded

# Initialize output and error files
def init_output_files():
    """Initialize the output and error files with headers."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize error file with timestamp header
    with open(ERROR_FILE, 'w') as f:
        f.write(f"# PDF Analysis Errors - {timestamp}\n\n")

def clean_json_string(json_str):
    """Clean up the JSON string by removing markdown code block markers."""
    if not isinstance(json_str, str):
        return json_str
        
    # Remove ```json and ``` markers
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)
    return json_str

def extract_json_objects(text):
    """Extract JSON objects from text that might contain multiple JSON blocks."""
    # First clean the text
    cleaned_text = clean_json_string(text)
    
    # Try to parse as a JSON array
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # If it's not a valid JSON array, try to extract individual JSON objects
        objects = []
        # Split on potential object boundaries when a new object starts
        parts = re.split(r'}\s*{', cleaned_text)
        
        for i, part in enumerate(parts):
            # Add missing braces for all but the first part
            if i > 0:
                part = '{' + part
            # Add missing braces for all but the last part
            if i < len(parts) - 1:
                part = part + '}'
            
            try:
                obj = json.loads(part.strip())
                objects.append(obj)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
                
        return objects

def write_to_error(error_msg):
    """Write an error message to the error file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_FILE, 'a') as f:
        f.write(f"[{timestamp}] {error_msg}\n\n")

def get_cache_key(filename, original_keywords):
    """
    Generate cache key with version to prevent mixing old/new analysis methods.
    Uses original keywords (not expanded) so adding keywords doesn't invalidate existing caches.
    """
    content = f"{CACHE_VERSION}:{filename}:{'|'.join(sorted(original_keywords))}"
    return hashlib.md5(content.encode()).hexdigest()

def check_cache(cache_key):
    """Check if analysis is cached and return it if available."""
    if NO_CACHE:
        return None
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_to_cache(cache_key, result):
    """Save analysis result to cache."""
    if NO_CACHE:
        return
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f"Warning: Could not cache results: {e}")
        write_to_error(f"Failed to cache results for key {cache_key}: {e}")

def quick_relevance_check(filename, text_sample, original_keywords):
    """
    Phase 2: Quick relevance check using the analysis model.
    Returns True if document might contain relevant themes, False otherwise.
    Cost: ~$0.0001 per document
    """
    keywords_str = ", ".join(original_keywords)
    prompt = f"""Does this document discuss ANY of these topics: {keywords_str}?

Consider the topics broadly - not just exact word matches, but related themes and concepts.

Answer ONLY with: YES or NO

Document excerpt:
{text_sample[:2000]}"""

    try:
        response = client.responses.create(
            model=ANALYSIS_MODEL,
            input=prompt,
            reasoning={"effort": "high"}
        )
        # Extract text from response
        answer = extract_text_from_response(response).strip().lower()
        return "yes" in answer
    except Exception as e:
        # On error, default to True (proceed with full analysis)
        print(f"Quick check failed for {filename}: {e}")
        return True

def analyze(filename, extracted_pages, original_keywords, expanded_keywords_dict):
    """
    Phase 3: Full semantic analysis of PDF file.
    Analyzes document for thematic relevance using expanded topic clusters.
    """
    cache_key = get_cache_key(filename, original_keywords)
    cached_result = check_cache(cache_key)
    if cached_result:
        return cached_result

    # Select representative pages for analysis
    selected_pages = select_content_for_analysis(extracted_pages)

    # Format topics compactly to reduce token usage
    topics_formatted = "\n".join([
        f"- {keyword}: {', '.join(topics[:10])}..."
        for keyword, topics in expanded_keywords_dict.items()
    ])

    # Format selected pages
    formatted_text = []
    for actual_page, text in selected_pages:
        # Truncate very long pages to manage token count
        truncated_text = text[:2000] if len(text) > 2000 else text
        formatted_text.append(f"Actual Page {actual_page}: {truncated_text}")

    document_text = "\n\n".join(formatted_text)

    system_prompt = f"""Analyze this PDF document for thematic relevance.

THEMES AND RELATED CONCEPTS:
{topics_formatted}

ANALYSIS REQUIREMENTS:
1. Determine if the document SUBSTANTIALLY discusses each theme
2. Use semantic understanding - match concepts and ideas, not just words
3. HIGH confidence requires:
   - Theme discussed across multiple paragraphs
   - 3+ related concepts present
   - Substantive discussion (not passing mentions)

OUTPUT RULES:
- Include ONLY high-confidence matches
- For each match, identify specific concepts found
- Extract 1-2 representative quotes with page numbers
- Return valid JSON in the exact format below

OUTPUT FORMAT:
{{
  "results": [
    {{
      "filename": "{filename}",
      "keyword": "keyword_name",
      "keyword_present": true,
      "confidence": "high",
      "matched_subtopics": ["concept1", "concept2"],
      "data": [
        {{
          "actual_page": "5",
          "text": "Relevant quote from document...",
          "subtopics_in_passage": ["concept1"]
        }}
      ]
    }}
  ]
}}

DOCUMENT TEXT (first pages and selected passages):
{document_text}

Return JSON only. Omit themes that don't meet HIGH confidence criteria."""

    for attempt in range(MAX_RETRIES):
        try:
            # Use semaphore to limit concurrent API calls
            with api_semaphore:
                response = client.responses.create(
                    model=ANALYSIS_MODEL,
                    input=system_prompt,
                    reasoning={"effort": "high"}
                )
                # Extract text from response
                result = extract_text_from_response(response)

            # Parse and extract the results array from the JSON object
            parsed_result = json.loads(result)
            results_array = parsed_result.get("results", [])

            # Filter to ensure only high-confidence results
            filtered_results = [
                r for r in results_array
                if r.get("keyword_present") and r.get("confidence") == "high" and r.get("data")
            ]

            # Save the filtered results array to cache
            save_to_cache(cache_key, filtered_results)

            return json.dumps(filtered_results)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"API call failed for {filename}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                error_msg = f"Failed to analyze {filename} after {MAX_RETRIES} attempts: {e}"
                print(error_msg)
                write_to_error(error_msg)
                return json.dumps({
                    "error": f"Analysis failed: {str(e)}",
                    "filename": filename
                })

def select_content_for_analysis(extracted_pages):
    """
    Select pages to send for semantic analysis.
    Strategy: First 2 pages + stratified sample of later pages for representative coverage.
    """
    selected_pages = []
    total_pages = len(extracted_pages)

    # Always include first 2 pages
    selected_pages.extend(extracted_pages[:min(2, total_pages)])

    # Add stratified sample from rest of document
    if total_pages > 2:
        # Sample pages at 25%, 50%, 75% and last page
        sample_indices = [
            int(total_pages * 0.25),
            int(total_pages * 0.50),
            int(total_pages * 0.75),
            total_pages - 1
        ]
        for idx in sample_indices:
            if idx >= 2:  # Don't duplicate first pages
                selected_pages.append(extracted_pages[idx])

    return selected_pages

def clean_text(text):
    """Clean text to handle hyphenation and improve readability."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Handle hyphenation (words split across lines with hyphens)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def preprocess_text_for_keywords(text, original_keywords):
    """
    Phase 1: Quick pre-filter using only original keywords.
    This is a rough gate - semantic analysis will do the real work.
    Fast string matching to filter obviously irrelevant documents.
    """
    text_lower = text.lower()

    for keyword in original_keywords:
        # Use word boundary for more accurate matching
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return True
        # Also do simple substring check in case word boundaries don't match
        if keyword.lower() in text_lower:
            return True

    return False

def extract_text_from_pdf_pikepdf(pdf_path):
    """
    Extract text from PDF using pikepdf (fastest for simple PDFs).
    Note: pikepdf is primarily for PDF manipulation, not text extraction.
    For complex PDFs with formatted text, PyMuPDF or pypdf work better.
    This implementation validates that text extraction worked properly.
    """
    extracted_pages = []

    try:
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            valid_text_found = False

            for page_num, page in enumerate(pdf.pages):
                actual_page_number = page_num + 1

                try:
                    # Try to extract text using pikepdf's basic method
                    text = ""
                    if "/Contents" in page:
                        contents = page.get("/Contents")
                        if contents:
                            # Handle both single content stream and array of streams
                            if not isinstance(contents, list):
                                contents = [contents]

                            for content_stream in contents:
                                if hasattr(content_stream, "read_bytes"):
                                    raw_text = content_stream.read_bytes().decode("latin-1", errors="ignore")
                                    text += raw_text

                    if text.strip():
                        clean_page_text = clean_text(text)
                        # Validate that we got actual readable text, not just PDF operators
                        # Check for PDF operators that indicate raw content stream
                        pdf_operators = ['/GS', '/CS', '/TT', 'BT', 'ET', 'Tm', 'TJ', 'Tf', ' gs', ' scn']
                        operator_count = sum(1 for op in pdf_operators if op in clean_page_text[:500])

                        # If we see too many PDF operators, this is not readable text
                        if operator_count > 5:
                            continue

                        if len(clean_page_text) > 10 and any(c.isalpha() for c in clean_page_text):
                            extracted_pages.append((actual_page_number, clean_page_text))
                            valid_text_found = True
                except Exception:
                    # Skip problematic pages
                    continue

                # Garbage collect if processing large PDFs
                if page_num % 50 == 0 and page_num > 0:
                    gc.collect()

            # If we didn't get any valid text, raise exception to trigger fallback
            if not valid_text_found:
                raise Exception("No valid text extracted from PDF")

    except Exception as e:
        raise Exception(f"pikepdf extraction failed: {e}")

    return extracted_pages

def extract_text_from_pdf_pymupdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    extracted_pages = []

    try:
        with fitz.open(pdf_path) as pdf_document:
            total_pages = pdf_document.page_count

            for page_num in range(total_pages):
                page = pdf_document[page_num]
                text = page.get_text("text")
                actual_page_number = page_num + 1

                if text.strip():
                    clean_page_text = clean_text(text)
                    extracted_pages.append((actual_page_number, clean_page_text))

                page = None

                if page_num % 50 == 0 and page_num > 0:
                    gc.collect()

    except Exception as e:
        raise Exception(f"PyMuPDF extraction failed: {e}")

    return extracted_pages

def extract_text_from_pdf_pypdf(pdf_path):
    """Extract text from PDF using pypdf."""
    extracted_pages = []

    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            total_pages = len(pdf.pages)

            for page_num in range(total_pages):
                page = pdf.pages[page_num]
                text = page.extract_text()
                actual_page_number = page_num + 1

                if text and text.strip():
                    clean_page_text = clean_text(text)
                    extracted_pages.append((actual_page_number, clean_page_text))

                if page_num % 50 == 0 and page_num > 0:
                    gc.collect()

    except Exception as e:
        raise Exception(f"pypdf extraction failed: {e}")

    return extracted_pages

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with fallback to multiple libraries."""
    extraction_methods = []

    # Build list of available extraction methods in order of performance
    if PDF_LIBRARY == "pikepdf" and AVAILABLE_PDF_LIBS.get('pikepdf'):
        extraction_methods.append(("pikepdf", extract_text_from_pdf_pikepdf))
    if AVAILABLE_PDF_LIBS.get('pymupdf'):
        extraction_methods.append(("pymupdf", extract_text_from_pdf_pymupdf))
    if AVAILABLE_PDF_LIBS.get('pypdf'):
        extraction_methods.append(("pypdf", extract_text_from_pdf_pypdf))

    if not extraction_methods:
        raise ImportError("No PDF extraction libraries available")

    # Try each method until one succeeds
    last_error = None
    for method_name, method_func in extraction_methods:
        try:
            result = method_func(pdf_path)
            if result:  # Successfully extracted text
                return result
        except Exception as e:
            last_error = e
            # Print fallback message if not the last method
            if method_name != extraction_methods[-1][0]:
                print(f"  {method_name} extraction failed, trying fallback...")
            continue

    # All methods failed
    error_msg = f"ERROR: Could not process {os.path.basename(pdf_path)}: {last_error}"
    print(error_msg)
    write_to_error(error_msg)
    return []

def process_pdf(pdf_path, original_keywords, expanded_keywords_dict):
    """
    Process a single PDF file with three-phase analysis.
    Phase 1: String pre-filter (fast, no cost)
    Phase 2: Quick relevance check (fast, minimal cost)
    Phase 3: Full semantic analysis (comprehensive, moderate cost)
    """
    filename = os.path.basename(pdf_path)

    try:
        extracted_pages = extract_text_from_pdf(pdf_path)

        if not extracted_pages:
            error_msg = f"WARNING: No text found in {filename}. Skipping analysis."
            print(error_msg)
            write_to_error(error_msg)
            return None

        # Combine all text
        all_text = "\n".join([text for _, text in extracted_pages])

        # Phase 1: Fast string-based pre-filter
        if not preprocess_text_for_keywords(all_text, original_keywords):
            message = f"  Phase 1: No keywords found in {filename}. Skipping."
            print(message)
            return json.dumps({
                "filename": filename,
                "message": "No keywords found in pre-filter",
                "phase": "1"
            })

        # Phase 2: Quick relevance check
        if not quick_relevance_check(filename, all_text, original_keywords):
            message = f"  Phase 2: Document not relevant to themes in {filename}. Skipping."
            print(message)
            return json.dumps({
                "filename": filename,
                "message": "Not relevant (quick check)",
                "phase": "2"
            })

        # Phase 3: Full semantic analysis
        print(f"  Phase 3: Analyzing {filename} for semantic themes...")
        result = analyze(filename, extracted_pages, original_keywords, expanded_keywords_dict)

        # Clear memory
        extracted_pages = None
        all_text = None
        gc.collect()

        return result
    except Exception as e:
        error_msg = f"ERROR in processing {filename}: {e}"
        print(error_msg)
        write_to_error(error_msg)
        return json.dumps({
            "filename": filename,
            "error": str(e)
        })

def load_progress():
    """Load progress from resume file if it exists."""
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    return {"processed_files": [], "results": []}

def save_progress(processed_files, results):
    """Save progress to resume file."""
    try:
        with open(RESUME_FILE, 'w') as f:
            json.dump({"processed_files": processed_files, "results": results}, f)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def process_pdfs_in_batches(pdf_directory, original_keywords, expanded_keywords_dict):
    """Process PDFs in batches with three-phase semantic analysis."""
    # Initialize output files
    init_output_files()

    # Load progress if exists
    progress = load_progress()
    processed_files = set(progress["processed_files"])
    all_results = progress["results"]

    # Track phase statistics
    phase_stats = {"phase_1_filtered": 0, "phase_2_filtered": 0, "phase_3_analyzed": 0}

    # Collect all PDF paths
    pdf_files = []
    for root, _, files in os.walk(pdf_directory):
        for filename in files:
            if filename.endswith('.pdf'):
                full_path = os.path.join(root, filename)
                if full_path not in processed_files:
                    pdf_files.append(full_path)

    if not pdf_files:
        print("No new PDF files to process.")
        if all_results:
            print(f"Found {len(all_results)} previously processed results.")
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {OUTPUT_FILE}")
        return

    total_files = len(pdf_files)
    print(f"\nFound {total_files} new PDF files to analyze")
    if processed_files:
        print(f"Resuming processing ({len(processed_files)} files already processed)")

    # Calculate batches
    batch_count = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        # Process each batch
        for batch_idx in range(batch_count):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, total_files)
            batch = pdf_files[start_idx:end_idx]

            print(f"\nProcessing batch {batch_idx+1}/{batch_count} (files {start_idx+1}-{end_idx}/{total_files})")

            batch_start_time = time.time()
            batch_results = []
            newly_processed = []

            # Process this batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_THREADS) as executor:
                future_to_pdf = {
                    executor.submit(process_pdf, pdf_path, original_keywords, expanded_keywords_dict): pdf_path
                    for pdf_path in batch
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_pdf), 
                                  total=len(batch),
                                  desc=f"Batch {batch_idx+1}/{batch_count}"):
                    pdf_path = future_to_pdf[future]
                    filename = os.path.basename(pdf_path)
                    
                    try:
                        result = future.result()
                        if result:
                            # Add to processed files list
                            newly_processed.append(pdf_path)

                            # Parse and add to results (skip phase filter messages)
                            try:
                                if isinstance(result, str):
                                    clean_result = clean_json_string(result)
                                    parsed_result = json.loads(clean_result)
                                else:
                                    parsed_result = result

                                # Track phase statistics
                                if isinstance(parsed_result, dict):
                                    if parsed_result.get("phase") == "1":
                                        phase_stats["phase_1_filtered"] += 1
                                        continue  # Skip adding to results
                                    elif parsed_result.get("phase") == "2":
                                        phase_stats["phase_2_filtered"] += 1
                                        continue  # Skip adding to results
                                    elif parsed_result.get("error"):
                                        # Keep error messages in results
                                        batch_results.append(parsed_result)
                                    else:
                                        # Unknown dict format, keep it
                                        batch_results.append(parsed_result)
                                elif isinstance(parsed_result, list):
                                    # Phase 3 analysis returns a list (even if empty)
                                    phase_stats["phase_3_analyzed"] += 1
                                    if parsed_result:  # Only add non-empty results
                                        batch_results.extend(parsed_result)
                                else:
                                    batch_results.append(parsed_result)
                            except Exception as e:
                                write_to_error(f"Error parsing result for {filename}: {e}")
                                objects = extract_json_objects(result)
                                if objects:
                                    if isinstance(objects, list):
                                        batch_results.extend(objects)
                                    else:
                                        batch_results.append(objects)
                    except Exception as e:
                        error_msg = f"\nError processing {filename}: {e}"
                        print(error_msg)
                        write_to_error(error_msg)
            
            # Add this batch's results to the main results
            all_results.extend(batch_results)
            processed_files.update(newly_processed)
            
            # Save intermediate results for this batch
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_idx+1} processed in {batch_time:.2f} seconds")
            
            # Save intermediate results
            with open(f"batch_{batch_idx+1}_results.json", 'w') as f:
                json.dump(batch_results, f, indent=2)
                
            # Save overall progress for resume capability
            save_progress(list(processed_files), all_results)
            
            # Force garbage collection between batches
            gc.collect()
        
        # Write all combined results at the end
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Clean up resume file if completed successfully
        if os.path.exists(RESUME_FILE):
            os.remove(RESUME_FILE)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted! Progress saved.")
        print(f"Processed {len(processed_files)} files so far.")
        print(f"Resume later by running the same command.")
        # Progress is already saved by the batch loop
    except Exception as e:
        error_msg = f"Error during processing: {e}"
        print(error_msg)
        write_to_error(error_msg)
        
        # Try to save what we have
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Errors logged to: {ERROR_FILE}")
    print(f"\nProcessing Statistics:")
    print(f"  - Total files processed: {len(processed_files)}")
    print(f"  - Phase 1 filtered (no keywords): {phase_stats['phase_1_filtered']}")
    print(f"  - Phase 2 filtered (not relevant): {phase_stats['phase_2_filtered']}")
    print(f"  - Phase 3 analyzed (full semantic): {phase_stats['phase_3_analyzed']}")
    print(f"  - Total keyword matches found: {len(all_results)}")

    # Calculate efficiency
    total_processed = len(processed_files)
    if total_processed > 0:
        phase_3_percentage = (phase_stats['phase_3_analyzed'] / total_processed) * 100
        print(f"\nCost Efficiency:")
        print(f"  - Only {phase_3_percentage:.1f}% of documents required full analysis")
        print(f"  - Estimated cost savings: ~{100 - phase_3_percentage:.0f}%")
    print(f"{'=' * 70}")

def process_pdfs_recursively(pdf_directory, original_keywords, expanded_keywords_dict):
    """Entry point function that calls the batch processing method."""
    process_pdfs_in_batches(pdf_directory, original_keywords, expanded_keywords_dict)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Update global variables with command line arguments
    pdf_directory = args.pdf_dir
    KEYWORDS_FILE = args.keywords_file
    OUTPUT_FILE = args.output_file
    ERROR_FILE = args.error_file
    EXPANSION_MODEL = args.expansion_model
    ANALYSIS_MODEL = args.analysis_model
    MAX_WORKERS_THREADS = args.max_workers
    MAX_API_CONCURRENCY = args.max_api_concurrency
    BATCH_SIZE = args.batch_size
    MAX_RETRIES = args.max_retries
    CACHE_DIR = args.cache_dir
    TOPICS_PER_KEYWORD = args.topics_per_keyword
    NO_CACHE = args.no_cache

    # Handle cache clearing
    if args.clear_cache:
        import shutil
        if os.path.exists(CACHE_DIR):
            print(f"Clearing cache directory: {CACHE_DIR}")
            shutil.rmtree(CACHE_DIR)
        if os.path.exists(EXPANDED_KEYWORDS_CACHE):
            print(f"Clearing keyword expansion cache: {EXPANDED_KEYWORDS_CACHE}")
            os.remove(EXPANDED_KEYWORDS_CACHE)

    # Ensure cache directory exists (unless --no-cache is used)
    if not args.no_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print("PDF Semantic Keyword Analysis Tool")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - PDF Directory: {pdf_directory}")
    print(f"  - Keywords File: {KEYWORDS_FILE}")
    print(f"  - Output File: {OUTPUT_FILE}")
    print(f"  - Error Log: {ERROR_FILE}")
    print(f"  - Expansion Model: {EXPANSION_MODEL} (with high reasoning effort)")
    print(f"  - Analysis Model: {ANALYSIS_MODEL}")
    print(f"  - Max Workers (Threads): {MAX_WORKERS_THREADS}")
    print(f"  - Max API Concurrency: {MAX_API_CONCURRENCY}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Max Retries: {MAX_RETRIES}")
    print(f"  - Cache Directory: {CACHE_DIR}")
    print(f"  - Caching: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"  - Topics Per Keyword: {TOPICS_PER_KEYWORD}")
    print(f"  - PDF Library: {PDF_LIBRARY}")
    print(f"  - Cache Version: {CACHE_VERSION}")
    print("=" * 70)
    print()

    # Load and expand keywords
    print("Loading keywords...")
    original_keywords, expanded_keywords_dict = load_or_expand_keywords(KEYWORDS_FILE)
    print(f"\nOriginal keywords: {', '.join(original_keywords)}")
    print()

    # Process PDFs with semantic analysis
    process_pdfs_recursively(pdf_directory, original_keywords, expanded_keywords_dict)
