Of course. This is a life-or-death situation, and every detail matters. The instructions will be broken down into a series of Markdown files, providing a complete, step-by-step operational guide.

Here is the file structure we will create:

```
UPSC_OCR_PROJECT/
├── 00_README.md
├── 01_SETUP.md
├── 02_PDF_PROCESSING.md
├── 03_THE_MASTER_PROMPT.md
├── 04_EXECUTION_SCRIPT.md
└── 05_VERIFICATION.md
```

---

### `00_README.md`

# URGENT: UPSC PDF to JSON Extraction Protocol

**SITUATION:** This is a high-stakes, time-sensitive operation. Failure is not an option.

**OBJECTIVE:** To accurately and rapidly OCR a PDF of UPSC previous year questions, classify each question by topic, and output the result as a structured JSON file.

**PRIMARY DIRECTIVE:** **Leverage, Don't Build.** Building a custom solution from scratch is too slow and error-prone. We will use state-of-the-art, commercially available AI models to achieve maximum speed and accuracy. The cost of API calls is irrelevant.

**STRATEGY: The Multimodal "God Mode" Approach**

We will employ a cutting-edge multimodal Large Language Model (LLM) like **OpenAI's GPT-4o**. This model can process images and text simultaneously. This allows us to collapse a complex, multi-step engineering pipeline (OCR -> Text Cleaning -> Parsing -> Classification -> JSON Formatting) into a single, powerful API call for each page of the PDF.

This is the fastest known method to achieve the objective.

### Operational Workflow

Follow these steps precisely. Each step is detailed in its corresponding file.

1.  **[Environment Setup (`01_SETUP.md`)](./01_SETUP.md)**: Prepare your computer, acquire API keys, and install necessary software.
2.  **[PDF Pre-processing (`02_PDF_PROCESSING.md`)](./02_PDF_PROCESSING.md)**: Convert the source PDF into a series of high-quality images.
3.  **[Prompt Engineering (`03_THE_MASTER_PROMPT.md`)](./03_THE_MASTER_PROMPT.md)**: Define the exact instructions for the AI to ensure it performs the task perfectly. This is the brain of the operation.
4.  **[Execution (`04_EXECUTION_SCRIPT.md`)](./04_EXECUTION_SCRIPT.md)**: Run the master script that sends the images to the AI and assembles the final JSON file.
5.  **[Verification (`05_VERIFICATION.md`)](./05_VERIFICATION.md)**: Quickly validate the output and deliver the result.

Proceed immediately to `01_SETUP.md`.

---

### `01_SETUP.md`

# Step 1: Environment Setup

**TIME ALLOTTED:** 5-7 minutes

**OBJECTIVE:** Prepare the system for the operation.

### Prerequisites

*   A powerful computer with a stable internet connection.
*   Python 3.8 or newer installed.
*   A code editor (e.g., VS Code, Sublime Text).
*   The source PDF file (e.g., `upsc_questions.pdf`).

### 1.1: Folder Structure

Create the following folder structure to keep the operation organized.

```
UPSC_OCR_PROJECT/
└── source_pdf/
    └── upsc_questions.pdf  <-- Place your PDF here
```

All your scripts will be created inside the `UPSC_OCR_PROJECT` directory.

### 1.2: Obtain OpenAI API Key

You need an API key to communicate with the GPT-4o model.

1.  Navigate to [platform.openai.com](https://platform.openai.com/).
2.  Log in or create an account immediately.
3.  Go to the "API Keys" section in the sidebar.
4.  Create a new secret key. **Copy it immediately and save it somewhere safe.** You will not be able to see it again.
5.  Ensure you have credits in your account. Go to the "Billing" section and add a payment method. Add at least $10-20 to be safe. The cost per page is low, but you need to ensure you don't hit a rate limit or run out of funds.

### 1.3: Set Up Python Environment

Working in a virtual environment is critical to avoid conflicts.

**Open your terminal or command prompt inside the `UPSC_OCR_PROJECT` directory.**

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

Your terminal prompt should now be prefixed with `(venv)`, indicating the virtual environment is active.

### 1.4: Install Dependencies

Install the required Python libraries using `pip`. These are the only tools you need.

```bash
pip install openai pymupdf tqdm
```

*   `openai`: The official Python client for the OpenAI API.
*   `pymupdf`: A high-performance library for PDF manipulation. We will use it to convert PDF pages to images.
*   `tqdm`: A utility that creates a smart progress bar. This is crucial for monitoring progress and managing stress.

**Setup is complete. Proceed immediately to `02_PDF_PROCESSING.md`.**

---

### `02_PDF_PROCESSING.md`

# Step 2: Convert PDF to High-Quality Images

**TIME ALLOTTED:** 3 minutes

**OBJECTIVE:** To convert each page of the source PDF into a high-resolution PNG image, which is the required input format for the multimodal AI.

**TOOL:** `PyMuPDF (fitz)` library.

### The Process

We will write a short Python script to perform this conversion. The script will:
1.  Open the source PDF.
2.  Create a directory named `pages` to store the output images.
3.  Iterate through each page of the PDF.
4.  Render each page as a high-resolution image (`300 DPI` is essential for OCR accuracy).
5.  Save the image with a numbered filename (e.g., `page_1.png`, `page_2.png`).

### The Script: `process_pdf.py`

Create a new file named `process_pdf.py` in your `UPSC_OCR_PROJECT` directory and paste the following code into it.

```python
import fitz  # PyMuPDF
import os
from tqdm import tqdm

# --- CONFIGURATION ---
PDF_PATH = os.path.join("source_pdf", "upsc_questions.pdf")
OUTPUT_DIR = "pages"
DPI = 300  # High DPI is CRITICAL for accurate OCR by the AI.

# --- SCRIPT LOGIC ---
def convert_pdf_to_images():
    """Opens the PDF and saves each page as a high-resolution PNG image."""
    print(f"Starting PDF conversion for: {PDF_PATH}")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    try:
        # Open the PDF document
        doc = fitz.open(PDF_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Could not open or read PDF file at {PDF_PATH}.")
        print(f"Error details: {e}")
        return

    # Iterate over each page with a progress bar
    print(f"Converting {len(doc)} pages to images at {DPI} DPI...")
    for i, page in enumerate(tqdm(doc, desc="Processing Pages")):
        # Get a pixmap (rasterized image) of the page
        pix = page.get_pixmap(dpi=DPI)
        
        # Define the output path for the image
        output_image_path = os.path.join(OUTPUT_DIR, f"page_{i + 1}.png")
        
        # Save the image
        pix.save(output_image_path)
    
    doc.close()
    print("\nPDF to image conversion complete.")
    print(f"All pages have been saved as PNG files in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    convert_pdf_to_images()

```

### Execution

1.  Make sure your PDF is located at `source_pdf/upsc_questions.pdf`.
2.  Run the script from your terminal (ensure your `venv` is still active):

    ```bash
    python process_pdf.py
    ```

3.  You will see a progress bar. Once complete, you will have a new folder named `pages` containing all the images.

**PDF processing is complete. Proceed immediately to `03_THE_MASTER_PROMPT.md`.**

---

### `03_THE_MASTER_PROMPT.md`

# Step 3: Engineering the "God Mode" Prompt

**TIME ALLOTTED:** 10 minutes

**OBJECTIVE:** To create a precise set of instructions (a "prompt") for the AI. A well-engineered prompt is the difference between success and failure. It forces the AI to behave predictably and deliver the exact output format we need.

### Principles of a "God Mode" Prompt

*   **Be Specific:** Tell the AI exactly what it is and what its goal is.
*   **Provide Structure:** Give it a numbered list of instructions.
*   **Define the Output Schema:** Show the AI the exact JSON structure you want. This is non-negotiable.
*   **Use Constraints:** Tell the AI what *not* to do (e.g., "Do not output any explanatory text").

### The Master Prompt

This is the prompt we will use. It has been carefully crafted for this specific task.

```text
You are an expert AI data extraction engine. Your sole purpose is to analyze an image of a page from a UPSC (Union Public Service Commission) exam question paper and convert its content into a structured JSON format.

Analyze the provided image. Identify every complete Multiple-Choice Question (MCQ) on the page. For each distinct MCQ you find, perform the following actions precisely:

1.  **EXTRACT DATA:**
    *   **Question Number:** Extract the question's identifier (e.g., "1.", "42.").
    *   **Question Text:** Extract the full, verbatim text of the question. Ensure you correctly handle multi-line questions and combine them into a single string.
    *   **Options:** Extract the text for all four options. Label them "a", "b", "c", and "d".

2.  **CLASSIFY TOPIC:**
    *   Based on the question's content, classify it into EXACTLY ONE of the following categories:
        - "Modern Indian History"
        - "Indian Polity and Governance"
        - "Economy and Social Development"
        - "Geography (India & World)"
        - "Science and Technology"
        - "Environment and Ecology"
        - "Art and Culture"
        - "Ancient & Medieval History"
        - "Miscellaneous"
    *   Use your knowledge to be as accurate as possible. For example, a question about the Indian Parliament belongs to "Indian Polity and Governance". A question about the Harappan civilization belongs to "Ancient & Medieval History".

3.  **FORMAT OUTPUT:**
    *   You MUST return a single, valid JSON array `[]`.
    *   Each object inside the array represents one question and MUST follow this exact schema:
    {
      "topic": "string (the category from the list above)",
      "question_number": "string",
      "question_text": "string",
      "options": {
        "a": "string",
        "b": "string",
        "c": "string",
        "d": "string"
      },
      "source_page": integer (This will be provided to you)
    }

**CRITICAL INSTRUCTIONS:**
*   If a page contains no questions (e.g., it's a cover page or blank), you MUST return an empty JSON array `[]`.
*   Your ENTIRE response must be ONLY the JSON array. Do NOT include any introductory text, explanations, apologies, or markdown formatting like ```json. Just the raw JSON.
```

There is no script to run in this step. You must understand this prompt, as it is the core logic that will be embedded in the final execution script.

**Prompt engineering is complete. Proceed immediately to `04_EXECUTION_SCRIPT.md`.**

---

### `04_EXECUTION_SCRIPT.md`

# Step 4: The Main Execution Script

**TIME ALLOTTED:** 5 minutes to code, then execution time.

**OBJECTIVE:** To create and run the master Python script that ties everything together. This script will:
1.  Read the Master Prompt.
2.  Find all the page images we created.
3.  Use parallel processing to send multiple images to the GPT-4o API simultaneously for maximum speed.
4.  Collect the JSON responses from the AI.
5.  Aggregate all responses into a single, final JSON file.

### The Script: `run_extraction.py`

Create a new file named `run_extraction.py` in your `UPSC_OCR_PROJECT` directory. Paste the following code into it.

**IMPORTANT:** Find the line `api_key="YOUR_API_KEY_HERE"` and replace `"YOUR_API_KEY_HERE"` with the key you obtained in Step 1.

```python
import os
import base64
import json
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY_HERE"  # !!! PASTE YOUR OPENAI API KEY HERE !!!
IMAGE_DIR = "pages"
OUTPUT_FILE = "final_upsc_questions.json"
MAX_WORKERS = 20  # Number of parallel API calls. Adjust based on your rate limits.

# --- LOAD THE MASTER PROMPT ---
# The prompt is loaded from the text provided in 03_THE_MASTER_PROMPT.md
MASTER_PROMPT = """
You are an expert AI data extraction engine. Your sole purpose is to analyze an image of a page from a UPSC (Union Public Service Commission) exam question paper and convert its content into a structured JSON format.

Analyze the provided image. Identify every complete Multiple-Choice Question (MCQ) on the page. For each distinct MCQ you find, perform the following actions precisely:

1.  **EXTRACT DATA:**
    *   **Question Number:** Extract the question's identifier (e.g., "1.", "42.").
    *   **Question Text:** Extract the full, verbatim text of the question. Ensure you correctly handle multi-line questions and combine them into a single string.
    *   **Options:** Extract the text for all four options. Label them "a", "b", "c", and "d".

2.  **CLASSIFY TOPIC:**
    *   Based on the question's content, classify it into EXACTLY ONE of the following categories:
        - "Modern Indian History"
        - "Indian Polity and Governance"
        - "Economy and Social Development"
        - "Geography (India & World)"
        - "Science and Technology"
        - "Environment and Ecology"
        - "Art and Culture"
        - "Ancient & Medieval History"
        - "Miscellaneous"
    *   Use your knowledge to be as accurate as possible. For example, a question about the Indian Parliament belongs to "Indian Polity and Governance". A question about the Harappan civilization belongs to "Ancient & Medieval History".

3.  **FORMAT OUTPUT:**
    *   You MUST return a single, valid JSON array `[]`.
    *   Each object inside the array represents one question and MUST follow this exact schema:
    {
      "topic": "string (the category from the list above)",
      "question_number": "string",
      "question_text": "string",
      "options": {
        "a": "string",
        "b": "string",
        "c": "string",
        "d": "string"
      },
      "source_page": integer
    }

**CRITICAL INSTRUCTIONS:**
*   If a page contains no questions (e.g., it's a cover page or blank), you MUST return an empty JSON array `[]`.
*   Your ENTIRE response must be ONLY the JSON array. Do NOT include any introductory text, explanations, apologies, or markdown formatting like ```json. Just the raw JSON.
"""

# --- HELPER FUNCTIONS ---
def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_page(image_file_path, page_number, client):
    """Sends a single page to the AI and returns the parsed JSON."""
    try:
        base64_image = encode_image_to_base64(image_file_path)
        
        # We replace the placeholder in the prompt with the actual page number
        prompt_for_page = MASTER_PROMPT.replace(
            "source_page\": integer", 
            f"source_page\": {page_number}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_for_page},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ],
                }
            ],
            temperature=0.0, # Set to 0 for maximum predictability and accuracy
            max_tokens=4000
        )
        
        response_content = response.choices[0].message.content
        
        # The AI is instructed to return only JSON, so we parse it directly.
        # Add a layer of safety by stripping potential markdown backticks.
        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()
        
        parsed_json = json.loads(response_content)
        return parsed_json

    except Exception as e:
        print(f"\nERROR processing page {page_number}: {e}")
        # Return an empty list on failure so the whole process doesn't stop.
        return []

# --- MAIN EXECUTION ---
def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("FATAL ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key in the script.")
        return

    client = OpenAI(api_key=API_KEY)
    
    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )

    if not image_files:
        print(f"FATAL ERROR: No images found in the '{IMAGE_DIR}' directory. Did you run `process_pdf.py`?")
        return

    all_questions = []
    
    print(f"Starting extraction of {len(image_files)} pages using up to {MAX_WORKERS} parallel workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(
                process_page, 
                img_path, 
                int(os.path.basename(img_path).split('_')[1].split('.')[0]), 
                client
            ): img_path for img_path in image_files
        }
        
        # Use tqdm to create a progress bar for the concurrent tasks
        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=len(image_files), desc="Extracting Questions"):
            page_questions = future.result()
            if page_questions:
                all_questions.extend(page_questions)

    print(f"\nExtraction complete. Total questions found: {len(all_questions)}")

    # Sort the final list by source page and then by question number
    # This assumes question numbers are extracted as strings that can be cast to int.
    all_questions.sort(key=lambda x: (x['source_page'], int(x.get('question_number', '0').rstrip('.'))))

    # Save the final aggregated list to a JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved all questions to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()

```

### Execution

1.  Confirm you have pasted your API key into the script.
2.  Run the script from your terminal:

    ```bash
    python run_extraction.py
    ```
3.  A progress bar will appear. This process may take several minutes, depending on the number of pages in the PDF. **Do not interrupt it.** The parallel processing is making it as fast as possible.
4.  Once finished, a file named `final_upsc_questions.json` will be created.

**The core operation is complete. Proceed immediately to `05_VERIFICATION.md` for the final step.**

---

### `05_VERIFICATION.md`

# Step 5: Final Verification and Delivery

**TIME ALLOTTED:** 5 minutes

**OBJECTIVE:** To perform a rapid but effective quality check on the final output before delivery.

### The Strategy: Spot-Checking

You do not have time to review every single entry. The goal is to verify the structural integrity and a sample of the data's accuracy.

1.  **Open `final_upsc_questions.json`** in your code editor. Most editors have built-in JSON formatting and validation.

2.  **Check 1: Structural Integrity.** Does the file open correctly? Is it a valid JSON array `[...]` containing objects `{...}`? If the script finished without errors, this should be fine.

3.  **Check 2: Data Spot-Check.** Open the original `upsc_questions.pdf` side-by-side with your JSON file. Randomly sample 3-5 data points from different parts of the document.

    *   **Pick a random page, e.g., page 25.** Find a question on that page in the PDF, e.g., question #48.
    *   **Search your JSON file** for `"source_page": 25` and `"question_number": "48"`.
    *   **Compare:**
        *   Is the `question_text` an exact match?
        *   Are the `options` ("a", "b", "c", "d") correct?
        *   Is the `topic` classification plausible? (e.g., Does a question about RBI monetary policy correctly get classified as "Economy and Social Development"?)

4.  **Repeat the spot-check** for another page, e.g., page 78, question #92.

### Interpreting Results

*   **Minor OCR errors:** A single misspelled word or a missed comma is acceptable. The AI is not perfect, but it's extremely good. This level of quality is sufficient.
*   **Major structural errors:** A missing option, a truncated question, or an incorrect JSON format. This is unlikely due to the prompt engineering but if found, you may need to re-run the process for that specific page.
*   **Topic misclassification:** This is the most likely "error". If a question is borderline, the AI might choose a different but plausible category. This is generally acceptable.

### Delivery

Once you have confirmed through spot-checking that the output is of high quality and structurally sound, the `final_upsc_questions.json` file is ready for delivery.

**OPERATION COMPLETE.**