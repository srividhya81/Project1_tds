# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "uv", 
#   "openai", 
#   "sqlite3", 
#   "requests", 
#   "npx",
#   "fastapi",
#   "python-dotenv",
#   "uvicorn",
#   "Pillow",
#   "pytesseract",
#   "markdown",
#   "pandas",
#   "duckdb",
#   "gitpython"
# ]
# ///   
from logging import config
import shutil
import pytesseract
from fastapi import FastAPI, Query
from typing import Dict, Any, Optional
import subprocess
import json
import sqlite3
import os
import re
import datetime
import sys
from collections import defaultdict
from openai import OpenAI
import requests
from dotenv import load_dotenv
import uvicorn
import git
import duckdb
import markdown
import pandas as pd

from PIL import Image

load_dotenv()


app = FastAPI()
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

DATA_DIR = "/Users/Project1_tds/data"  # Ensure this is writable
os.makedirs(DATA_DIR, exist_ok=True)  # Create it if it doesn't exist

def run_datagen(user_email: str):
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    script_path = "datagen.py"

    try:
        # Download the script
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the script locally
        with open(script_path, "w") as f:
            f.write(response.text)

        # Modify the script to change the root directory dynamically
        with open(script_path, "r") as f:
            script_content = f.read()
        
        # Replace '/data' with our custom writable directory
        script_content = script_content.replace('f"{DATA_DIR}"', f'"{DATA_DIR}"')

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make sure the script is executable
        os.chmod(script_path, 0o755)

        # Execute the script with user_email as an argument
        subprocess.run(["python3", script_path, user_email], check=True)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the script: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")
    finally:
        # Cleanup: Remove the script after execution
        if os.path.exists(script_path):
            os.remove(script_path)


def format_file():
    subprocess.run(["npx", "prettier@3.4.2", "--write", f"{DATA_DIR}/format.md"])



def count_weekday(weekday: str):
    with open(f"{DATA_DIR}/dates.txt") as f:
        dates = []
        for line in f:
            try:
                date_obj = datetime.datetime.strptime(line.strip(), "%Y-%m-%d")
            except ValueError:
                try:
                    date_obj = datetime.datetime.strptime(line.strip(), "%d-%b-%Y")
                except ValueError:
                    try:
                        date_obj = datetime.datetime.strptime(line.strip(), "%Y/%m/%d %H:%M:%S")
                    except ValueError:
                        continue
            dates.append(date_obj.weekday())
    
    weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    count = sum(1 for day in dates if day == weekday_map[weekday.lower()])
    with open(f"{DATA_DIR}/dates-{weekday.lower()}.txt", "w") as f:
        f.write(str(count))


def sort_contacts(order: str = "asc", sort_by: str = "first_name"):
    with open(f"{DATA_DIR}/contacts.json") as f:
        contacts = json.load(f)
    reverse = order.lower() == "desc"
    sorted_contacts = sorted(contacts, key=lambda c: c[sort_by], reverse=reverse)
    with open(f"{DATA_DIR}/contacts-sorted.json", "w") as f:
        json.dump(sorted_contacts, f, indent=2)

def extract_logs(time_period: str = "recent"):
    time_period_map = {"new": "recent", "past": "old"}
    time_period = time_period_map.get(time_period.lower(), time_period)
    
    log_files = sorted([f for f in os.listdir(f"{DATA_DIR}/logs") if f.endswith(".log")],
                       key=lambda x: os.path.getmtime(f"{DATA_DIR}/logs/{x}"),
                       reverse=(time_period == "recent"))
    if time_period == "recent":
        log_files = log_files[:10]
    elif time_period == "old":
        log_files = log_files[-10:]
    with open(f"{DATA_DIR}/logs-{time_period}.txt", "w") as f:
        for log in log_files:
            with open(f"{DATA_DIR}/logs/{log}") as lf:
                f.write(lf.readline())

def generate_markdown_index():
    index = {}
    for root, _, files in os.walk(f"{DATA_DIR}/docs/"):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file] = line.strip("# ").strip()
                            break
    with open(f"{DATA_DIR}/docs/index.json", "w") as f:
        json.dump(index, f, indent=2)


def extract_email_sender(query: str):
    with open(f"{DATA_DIR}/email.txt") as f:
        email_content = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract sender name, receiver email ID, receiver name, subject, date, and CC from the following email. Return the result strictly as a JSON object with keys: sender_name, receiver_email, receiver_name, subject, date, and cc."},
            {"role": "user", "content": email_content}
        ]
    )

    response_content = response.choices[0].message.content.strip()
    
    # Debugging: Print response content
    print("LLM Response:", response_content)

    try:
        extracted_info = json.loads(response_content)
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from LLM. Attempting to extract key-value pairs manually.")
        extracted_info = {}
        
        # Simple manual extraction in case JSON decoding fails
        lines = response_content.split("\n")
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                extracted_info[key.strip()] = value.strip()

    # Determine which fields to include based on the query
    requested_fields = []
    field_map = {
        "sender": "sender_name",
        "receiver": "receiver_email",
        "receiver name": "receiver_name",
        "subject": "subject",
        "date": "date",
        "cc": "cc"
    }

    for key in field_map:
        if key in query.lower():
            requested_fields.append(field_map[key])

    if not requested_fields:
        requested_fields = field_map.values()  # Default to all fields if none are explicitly asked

    # Save only the requested fields in a text file
    with open(f"{DATA_DIR}/email-sender.txt", "w") as f:
        for field in requested_fields:
            if field in extracted_info:
                f.write(f"{field.replace('_', ' ').title()}: {extracted_info[field]}\n")

def extract_text_from_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

def parse_credit_card_info(text: str) -> Dict[str, Optional[str]]:
    card_number_match = re.search(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b", text)
    valid_thru_match = re.search(r"\b(0[1-9]|1[0-2])/(\d{2})\b", text)
    cvv_matches = re.findall(r"\b\d{3}\b", text)
    cvv_match = cvv_matches[0] if cvv_matches else None  # Ensure only the first valid 3-digit CVV is taken
    
    # Exclude words like "VALID" and "THRU" from being detected as names
    name_candidates = re.findall(r"\b[A-Z][A-Z ]{2,}\b", text)
    filtered_names = [name for name in name_candidates if name not in {"VALID", "THRU"}]
    name_match = filtered_names[-1] if filtered_names else None  # Take the last detected name, assuming it appears at the end

    return {
        "credit_card_number": card_number_match.group(0) if card_number_match else None,
        "name": name_match,
        "valid_thru": valid_thru_match.group(0) if valid_thru_match else None,
        "cvv": cvv_match,
    }

def extract_credit_card_info(query: str):
    image_path = f"{DATA_DIR}/credit_card.png"
    extracted_text = extract_text_from_image(image_path)
    
    if not extracted_text.strip():
        return {"error": "No text extracted from the image."}
    
    extracted_info = parse_credit_card_info(extracted_text)

    # Determine which fields to include based on the query
    requested_fields = []
    field_map = {
        "card number": "credit_card_number",
        "name": "name",
        "valid thru": "valid_thru",
        "cvv": "cvv"
    }

    for key in field_map:
        if key in query.lower():
            requested_fields.append(field_map[key])

    if not requested_fields:
        requested_fields = field_map.values()  # Default to all fields if none are explicitly asked

    # Save only the requested fields in a text file
    output_file = os.path.join(DATA_DIR, "credit-card.txt")
    try:
        with open(output_file, "w") as f:
            for field in requested_fields:
                if extracted_info[field]:
                    f.write(f"{field}: {extracted_info[field]}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

    return {field: extracted_info[field] for field in requested_fields}


def find_similar_comments():
    with open(f"{DATA_DIR}/comments.txt") as f:
        comments = f.readlines()
    response = client.embeddings.create(model="text-embedding-3-small", input=comments)
    embeddings = {comment: emb.embedding for comment, emb in zip(comments, response.data)}
    most_similar = min(embeddings.keys(), key=lambda x: sum(embeddings[x]))
    with open(f"{DATA_DIR}/comments-similar.txt", "w") as f:
        f.writelines(most_similar)

def calculate_ticket_sales(ticket_type: Optional[str] = None):
    conn = sqlite3.connect(f"{DATA_DIR}/ticket-sales.db")
    cur = conn.cursor()
    if ticket_type:
        cur.execute("SELECT type, SUM(units * price) FROM tickets WHERE LOWER(type) = LOWER(?) GROUP BY type", (ticket_type,))
    else:
        cur.execute("SELECT type, SUM(units * price) FROM tickets GROUP BY type")
    sales = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    with open(f"{DATA_DIR}/ticket-sales.txt", "w") as f:
        json.dump(sales, f, indent=2)

def map_query_to_task(query: str):
    prompt = """You are an AI that maps user queries to predefined function names. 
    Given a query, return the best matching function from this list: 

    - run_datagen: Install dependencies and run data generation
    - format_file: Format a file using a specified formatter
    - count_weekday: Count occurrences of a specific weekday in a file
    - sort_contacts: Sort contacts by first and last name in ascending or descending order
    - extract_logs: Extract logs based on a given time range (recent, old, or specific period),recent can also be new old can also be past like relatable words should be considered
    - generate_markdown_index: Create an index from Markdown files
    - extract_email_sender: Extract the sender's email from an email file
    - extract_credit_card_info: Extract a credit card number, name, valid thru and cvv number from an image
    - find_similar_comments: Identify the most similar comments using embeddings
    - calculate_ticket_sales: Compute total ticket sales, optionally filtered by ticket type. the ticket type may be specified in query itself like gold ticket sale or silver ticket sales consider that also

    Respond with only the function name, without extra explanations.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()


TASKS = {
    "run_datagen": run_datagen,
    "format_file": format_file,
    "count_weekday": count_weekday,
    "sort_contacts": sort_contacts,
    "extract_logs": extract_logs,
    "generate_markdown_index": generate_markdown_index,
    "extract_email_sender": extract_email_sender,
    "extract_credit_card_info": extract_credit_card_info,
    "find_similar_comments": find_similar_comments,
    "calculate_ticket_sales": calculate_ticket_sales,
}

@app.get("/run")
def run_task(query: str, param: Optional[str] = Query(None)):
    task = map_query_to_task(query)
    if task not in TASKS:
        return {"error": "Invalid task"}
    if param:
        TASKS[task](param)
    else:
        TASKS[task]()
    return {"status": "Task executed successfully"}


# Additional Automation Tasks
def check_path_security(path: str):
    if not path.startswith(DATA_DIR):
        raise PermissionError("Access to files outside /data is not allowed.")

def safe_open(file_path: str, mode: str):
    check_path_security(file_path)
    return open(file_path, mode)

@app.get("/fetch-api-data")
def fetch_api_data(url: str, filename: str):
    response = requests.get(url)
    check_path_security(os.path.join(DATA_DIR, filename))
    with safe_open(os.path.join(DATA_DIR, filename), "w") as f:
        f.write(response.text)
    return {"status": "Data fetched successfully", "filename": filename}

@app.get("/clone-git-repo")
def clone_git_repo(repo_url: str, folder_name: str):
    check_path_security(os.path.join(DATA_DIR, folder_name))
    git.Repo.clone_from(repo_url, os.path.join(DATA_DIR, folder_name))
    return {"status": "Repository cloned successfully", "folder": folder_name}

@app.get("/run-sql-query")
def run_sql_query(db_name: str, query: str):
    check_path_security(os.path.join(DATA_DIR, db_name))
    conn = duckdb.connect(os.path.join(DATA_DIR, db_name))
    result = conn.execute(query).fetchall()
    return {"query_result": result}

@app.get("/compress-image")
def compress_image(input_path: str, output_path: str):
    check_path_security(input_path)
    check_path_security(output_path)
    img = Image.open(input_path)
    img.save(output_path, "JPEG", quality=50)
    return {"status": "Image compressed successfully", "output_path": output_path}

@app.get("/transcribe-audio")
def transcribe_audio(file_path: str):
    check_path_security(file_path)
    return {"transcription": "Transcription output"}  # Placeholder for actual transcription logic

@app.get("/convert-markdown")
def convert_markdown_to_html(md_content: str):
    return {"html": markdown.markdown(md_content)}

@app.get("/filter-csv")
def filter_csv(file_path: str, column: str, value: str):
    check_path_security(file_path)
    df = pd.read_csv(file_path)
    return {"filtered_data": df[df[column] == value].to_json()}

# Block deletion functions to prevent accidental deletions
def restricted_os_remove(path):
    raise PermissionError("File deletion is not allowed.")

def restricted_shutil_rmtree(path):
    raise PermissionError("Directory deletion is not allowed.")

os.remove = restricted_os_remove
shutil.rmtree = restricted_shutil_rmtree


if __name__ == "__main__":
    uvicorn.run(app,port = 8001)