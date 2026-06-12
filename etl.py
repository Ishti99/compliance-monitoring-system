import os
import boto3
import json

def get_bedrock_client():
    session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
    return session.client("bedrock-runtime", region_name="us-west-2")

# read raw text from a file
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            print("pdfplumber not installed, trying basic read...")
            with open(filepath, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")

    else:
        # just try reading it as text
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# uses llm to extract structured findings from raw audit text
def extract_findings(raw_text):
    client = get_bedrock_client()

    prompt = f"""You are a labor compliance analyst.
Read the following factory audit text and extract key findings.
For each factory mentioned, list:
- Factory name
- Violations found
- Compliance status

Return the findings in a clear structured format.

Audit text:
{raw_text}"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = get_bedrock_client().invoke_model(
        body=body,
        modelId="deepseek.v3.2",
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get("body").read())
    return result["choices"][0]["message"]["content"]

# save extracted text back to a file for embedding later
def save_extracted_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved extracted text to {output_path}")