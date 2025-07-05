import os
import re
import base64
import json
import concurrent.futures
import shutil
import fitz  # PyMuPDF
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
SOURCE_PDF = "source_pdf/upsc_questions.pdf"
IMAGE_DIR = "pages"
OUTPUT_FILE = "final_upsc_questions_classified.json"
MAX_WORKERS = 20
TEMP_DIR = "temp_results_classified"
TOPIC_FILES = [
    "topics_paper_1.md",
    "topics_paper_2.md"
]

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
You are an expert AI data extraction engine. Your purpose is to analyze a page from a UPSC exam question paper, extract all questions, and associate them with any diagrams present.

**CONTEXT:**
The automated system has pre-extracted all images from this page. They are identified by a `diagram_id`.
{diagram_list}

**YOUR TASK:**
Analyze the full page image provided. For each Multiple-Choice Question (MCQ) you find, perform the following actions:

1.  **EXTRACT DATA & ASSOCIATE DIAGRAM:**
    *   **Question Number:** Extract the question's identifier (e.g., "1.", "42.").
    *   **Question Text:** Extract the full, verbatim text of the question.
    *   **Options:** Extract the text for all four options ("a", "b", "c", "d").
    *   **Diagram Association:** If the question refers to a diagram, identify its corresponding `diagram_id` from the context I provided. If there is no associated diagram, this value must be `null`.

2.  **CLASSIFY TOPIC:**
    *   Based on the question's content, classify it into EXACTLY ONE of the following categories:
{topic_list}
    *   If no other category fits, use "Miscellaneous".

3.  **FORMAT OUTPUT:**
    *   You MUST return a single, valid JSON array `[]`.
    *   Each object in the array represents one question and MUST follow this exact schema:
    {{
      "topic": "string",
      "question_number": "string",
      "question_text": "string",
      "options": {{
        "a": "string",
        "b": "string",
        "c": "string",
        "d": "string"
      }},
      "diagram_id": "integer or null",
      "source_page": {source_page_number}
    }}

**CRITICAL INSTRUCTIONS:**
*   If a page contains no questions, return an empty JSON array `[]`.
*   Your ENTIRE response must be ONLY the JSON array. Do NOT include any introductory text, explanations, or markdown formatting like ```json. Just the raw JSON.
"""

# --- PDF PROCESSING ---
def convert_pdf_to_images(pdf_path, image_dir):
    """Converts a PDF to PNG images, one for each page, skipping existing ones."""
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"Created directory: {image_dir}")

    try:
        doc = fitz.open(pdf_path)
        print(f"Processing PDF '{pdf_path}' with {len(doc)} pages.")
        
        for page_num in tqdm(range(len(doc)), desc="Converting PDF to Images"):
            page_index = page_num + 1
            image_path = os.path.join(image_dir, f"page_{page_index}.png")
            
            if os.path.exists(image_path):
                continue  # Skip if image already exists
                
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # High DPI for better OCR quality
            pix.save(image_path)
            
        print("PDF to image conversion complete.")
        doc.close()
    except Exception as e:
        print(f"ERROR converting PDF to images: {e}")
        return False
    return True

# --- HELPER FUNCTIONS ---
def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_integer_from_string(s):
    """Extracts the first integer from a string, returns 0 if not found."""
    if not isinstance(s, str):
        return 0
    match = re.search(r'\d+', s)
    if match:
        return int(match.group(0))
    return 0

# --- TOPIC LOADING ---
def load_topics_from_md(file_path):
    """Reads a Markdown file and extracts topics, filtering out noise."""
    if not os.path.exists(file_path):
        print(f"WARNING: Topic file not found at {file_path}. Skipping.")
        return []
    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Filter out common noise, headers, and footers
            if not line or line.startswith('#') or "GS SCORE" in line or "www.iasscore.in" in line or "---" in line or line.startswith('™'):
                continue
            # Remove list markers and other non-topic text
            cleaned_line = re.sub(r'^\s*[-*+]\s*|\d+\.\s*', '', line).strip()
            if cleaned_line:
                topics.append(cleaned_line)
    return topics

# --- AI PROCESSING WITH FAILSAFE ---
def process_page_for_classification(pdf_path, page_number, model, master_prompt_template):
    """Extracts text and diagrams from a single PDF page and saves the result."""
    temp_file_path = os.path.join(TEMP_DIR, f"page_{page_number}.json")
    if os.path.exists(temp_file_path):
        return  # Skip if already processed

    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_number - 1)

            # 1. Pre-extract all images from the page
            diagram_candidates = []
            for i, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                diagram_candidates.append({
                    "id": i,
                    "base64": base64.b64encode(image_bytes).decode('utf-8')
                })

            # 2. Create the context for the prompt
            if diagram_candidates:
                diagram_list_for_prompt = f"This page contains {len(diagram_candidates)} diagrams, identified by `diagram_id` from 0 to {len(diagram_candidates)-1}."
            else:
                diagram_list_for_prompt = "No diagrams were extracted from this page."

            # 3. Get the full page image for visual context
            pix = page.get_pixmap(dpi=300)
            full_page_b64 = base64.b64encode(pix.tobytes("png")).decode('utf-8')
            image_part = {"mime_type": "image/png", "data": full_page_b64}

            # 4. Format the final prompt for this page
            prompt_for_page = master_prompt_template.format(
                diagram_list=diagram_list_for_prompt,
                topic_list="{topic_list}", # Keep this as a placeholder
                source_page_number=page_number
            )

            # 5. Call the AI model
            response = model.generate_content(
                [prompt_for_page, image_part],
                generation_config={"temperature": 0.0, "max_output_tokens": 4096}
            )

            # 6. Clean and parse the AI's response
            response_content = response.text.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            
            parsed_json = json.loads(response_content)

            # 7. Assemble the final JSON, embedding the diagram Base64
            final_page_questions = []
            for question in parsed_json:
                diagram_id = question.pop("diagram_id", None)
                question["diagram_base64"] = None
                if diagram_id is not None and 0 <= diagram_id < len(diagram_candidates):
                    question["diagram_base64"] = diagram_candidates[diagram_id]["base64"]
                final_page_questions.append(question)

            # 8. Save the enriched data to a temporary file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_page_questions, f)

    except Exception as e:
        print(f"\nERROR processing page {page_number}: {e}")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

# --- MAIN EXECUTION ---
def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("FATAL ERROR: GOOGLE_API_KEY not found. Please create a .env file and add your key.")
        return
    
    genai.configure(api_key=google_api_key)

    # Step 1: Convert PDF to images
    if not convert_pdf_to_images(SOURCE_PDF, IMAGE_DIR):
        return  # Stop if conversion fails

    # Step 2: Failsafe Setup
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"--- Created temporary directory '{TEMP_DIR}' for failsafe results. ---")
    else:
        print(f"--- Found existing temporary directory '{TEMP_DIR}'. Resuming... ---")

    # Step 3: Load and combine topics
    all_topics = []
    for file_path in TOPIC_FILES:
        all_topics.extend(load_topics_from_md(file_path))
    
    if not all_topics:
        print("FATAL ERROR: No topics were loaded. Check topic file paths and content.")
        return
        
    unique_topics = sorted(list(set(all_topics)))
    if "Miscellaneous" not in unique_topics:
        unique_topics.append("Miscellaneous")
    
    print(f"--- Loaded and Combined {len(unique_topics)} Topics ---")
    
    # Step 4: Build the master prompt
    topic_list_for_prompt = "\n".join([f"        - \"{topic}\"" for topic in unique_topics])
    master_prompt = PROMPT_TEMPLATE.format(topic_list=topic_list_for_prompt, source_page_number="{source_page_number}")

    # Step 5: Proceed with extraction
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")],
        key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
    )

    if not image_files:
        print(f"FATAL ERROR: No images found in '{IMAGE_DIR}'.")
        return
    
    print(f"\nStarting classification of {len(image_files)} pages using up to {MAX_WORKERS} parallel workers...")
    
    # We pass the master_prompt template and let the worker format it with page-specific details
    master_prompt_with_topics = master_prompt.replace("{topic_list}", topic_list_for_prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # The executor now iterates over page numbers, not image files
        page_numbers = range(1, len(image_files) + 1)
        future_to_page = {
            executor.submit(
                process_page_for_classification,
                SOURCE_PDF,
                page_num,
                model,
                master_prompt_with_topics
            ): page_num for page_num in page_numbers
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=len(page_numbers), desc="Classifying Questions"):
            future.result()  # Check for exceptions

    print("\n--- All pages processed. Consolidating results. ---")

    # Step 6: Failsafe Aggregation
    all_questions = []
    temp_files = os.listdir(TEMP_DIR)
    for temp_file in tqdm(temp_files, desc="Consolidating"):
        temp_file_path = os.path.join(TEMP_DIR, temp_file)
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                page_questions = json.load(f)
                if page_questions:
                    all_questions.extend(page_questions)
        except (json.JSONDecodeError, IOError) as e:
            print(f"\nWarning: Could not read or parse temp file {temp_file}: {e}")

    print(f"\nClassification complete. Total questions found: {len(all_questions)}")
    
    # Step 7: Sort and Save
    all_questions.sort(key=lambda x: (x.get('source_page', 0), extract_integer_from_string(x.get('question_number', '0'))))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved all classified questions to '{OUTPUT_FILE}'.")

    # Step 8: Failsafe Cleanup
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"--- Successfully removed temporary directory '{TEMP_DIR}'. ---")
    except OSError as e:
        print(f"Error: Could not remove temporary directory {TEMP_DIR}: {e}")

if __name__ == "__main__":
    main()