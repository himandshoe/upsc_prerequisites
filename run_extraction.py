import os
import re
import base64
import json
import concurrent.futures
import shutil
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
IMAGE_DIR = "pages"
OUTPUT_FILE = "final_upsc_questions.json"
MAX_WORKERS = 20
TEMP_DIR = "temp_results"  # Directory for intermediate results

# --- NEW: Paths to your topic Markdown files ---
TOPIC_FILES = [
    "topics_paper_1.md",
    "topics_paper_2.md"
]

# --- NEW: Function to load topics from a Markdown file ---
def load_topics_from_md(file_path):
    """
    Reads a Markdown file and extracts topics.
    Assumes topics are list items (e.g., "- Topic Name") or one topic per line.
    """
    if not os.path.exists(file_path):
        print(f"WARNING: Topic file not found at {file_path}. Skipping.")
        return []

    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Clean the line: strip whitespace and remove markdown list markers
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and headers
                continue
            
            # Use regex to remove common list markers like '-', '*', '+', or '1.'
            cleaned_line = re.sub(r'^\s*[-*+]\s*|\d+\.\s*', '', line)
            topics.append(cleaned_line.strip())
            
    return topics

# --- MODIFIED: Prompt is now a template ---
PROMPT_TEMPLATE = """
You are an expert AI data extraction engine. Your sole purpose is to analyze an image of a page from a UPSC (Union Public Service Commission) exam question paper and convert its content into a structured JSON format.

Analyze the provided image. Identify every complete Multiple-Choice Question (MCQ) on the page. For each distinct MCQ you find, perform the following actions precisely:

1.  **EXTRACT DATA:**
    *   **Question Number:** Extract the question's identifier (e.g., "1.", "42.").
    *   **Question Text:** Extract the full, verbatim text of the question. Ensure you correctly handle multi-line questions and combine them into a single string.
    *   **Options:** Extract the text for all four options. Label them "a", "b", "c", and "d".

2.  **CLASSIFY TOPIC:**
    *   Based on the question's content, classify it into EXACTLY ONE of the following categories:
{topic_list}
    *   Use your knowledge to be as accurate as possible. If no other category fits, use "Miscellaneous".

3.  **FORMAT OUTPUT:**
    *   You MUST return a single, valid JSON array `[]`.
    *   Each object inside the array represents one question and MUST follow this exact schema:
    {{
      "topic": "string (the category from the list above)",
      "question_number": "string",
      "question_text": "string",
      "options": {{
        "a": "string",
        "b": "string",
        "c": "string",
        "d": "string"
      }},
      "source_page": {source_page_number}
    }}

**CRITICAL INSTRUCTIONS:**
*   If a page contains no questions (e.g., it's a cover page or blank), you MUST return an empty JSON array `[]`.
*   Your ENTIRE response must be ONLY the JSON array. Do NOT include any introductory text, explanations, apologies, or markdown formatting like ```json. Just the raw JSON.
"""

# --- HELPER FUNCTIONS (unchanged) ---
def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- MODIFIED: process_page now accepts the master_prompt as an argument ---
def process_page(image_file_path, page_number, model, master_prompt):
    """Sends a single page to the AI, saves the result to a temp file."""
    temp_file_path = os.path.join(TEMP_DIR, f"page_{page_number}.json")
    if os.path.exists(temp_file_path):
        return  # Skip if already processed

    try:
        base64_image = encode_image_to_base64(image_file_path)
        
        image_part = {
            "mime_type": "image/png",
            "data": base64_image
        }
        
        prompt_for_page = master_prompt.replace("{source_page_number}", str(page_number))

        response = model.generate_content(
             [prompt_for_page, image_part],
             generation_config={
                "temperature": 0.0,
                "max_output_tokens": 4096
             }
        )
        
        response_content = response.text
        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()
        
        parsed_json = json.loads(response_content)
        
        # Failsafe: Save the result to a temporary file
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f)

    except Exception as e:
        print(f"\nERROR processing page {page_number}: {e}")
        # Create an empty file to mark as processed (with error)
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

def extract_integer_from_string(s):
    """Extracts the first integer from a string, returns 0 if not found."""
    if not isinstance(s, str):
        return 0
    match = re.search(r'\d+', s)
    if match:
        return int(match.group(0))
    return 0

# --- MODIFIED: Main execution block to build the prompt dynamically ---
def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("FATAL ERROR: GOOGLE_API_KEY not found. Please create a .env file and add your key.")
        return
    
    genai.configure(api_key=google_api_key)

    # --- Failsafe Setup ---
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"--- Created temporary directory '{TEMP_DIR}' for failsafe results. ---")
    else:
        print(f"--- Found existing temporary directory '{TEMP_DIR}'. Resuming... ---")

    # Step 1: Load topics from all specified files
    all_topics = []
    for file_path in TOPIC_FILES:
        all_topics.extend(load_topics_from_md(file_path))
    
    # Step 2: De-duplicate topics and add a catch-all
    if not all_topics:
        print("FATAL ERROR: No topics were loaded from your MD files. Please check the file paths and content.")
        return
        
    unique_topics = sorted(list(set(all_topics)))
    if "Miscellaneous" not in unique_topics:
        unique_topics.append("Miscellaneous") # Safety net
    
    print("--- Loaded and Combined Topics ---")
    for topic in unique_topics:
        print(f"- {topic}")
    print("---------------------------------")
    
    # Step 3: Build the final master prompt
    topic_list_for_prompt = "\n".join([f"        - \"{topic}\"" for topic in unique_topics])
    master_prompt = PROMPT_TEMPLATE.format(topic_list=topic_list_for_prompt, source_page_number="{source_page_number}")

    # Step 4: Proceed with extraction
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )

    if not image_files:
        print(f"FATAL ERROR: No images found in '{IMAGE_DIR}'. Did you run the PDF processing script?")
        return
    
    print(f"\nStarting extraction of {len(image_files)} pages using up to {MAX_WORKERS} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(
                process_page,
                img_path,
                int(os.path.basename(img_path).split('_')[1].split('.')[0]),
                model,
                master_prompt
            ): img_path for img_path in image_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=len(image_files), desc="Extracting Questions"):
            future.result()  # Check for exceptions

    print("\n--- All pages processed. Consolidating results. ---")

    # --- Failsafe Aggregation ---
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

    print(f"\nExtraction complete. Total questions found: {len(all_questions)}")
    all_questions.sort(key=lambda x: (x.get('source_page', 0), extract_integer_from_string(x.get('question_number', '0'))))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved all questions to '{OUTPUT_FILE}'.")

    # --- Failsafe Cleanup ---
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"--- Successfully removed temporary directory '{TEMP_DIR}'. ---")
    except OSError as e:
        print(f"Error: Could not remove temporary directory {TEMP_DIR}: {e}")


if __name__ == "__main__":
    main()