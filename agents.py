import os
import boto3
import json
from rag import search, build_context

os.environ["AWS_PROFILE"] = "GSB570-BedrockOnly-490332585640"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

_df = None

def set_dataframe(df):
    global _df
    _df = df

def get_bedrock_client():
    from botocore.config import Config
    session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
    return session.client("bedrock-runtime", region_name="us-west-2",
        config=Config(read_timeout=120))

def call_llm(prompt, max_tokens=2000):
    client = get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    })
    response = client.invoke_model(
        body=body,
        modelId="deepseek.v3.2",
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get("body").read())
    return result["choices"][0]["message"]["content"]

# quick answer - just returns a clean list of factories with issues
def run_single_agent(question):
    if _df is None:
        return "no data loaded"
    results = search(_df, question, top_n=80, min_score=0.35)
    context = build_context(results)
    prompt = f"""You are a compliance analyst reviewing garment factory audit reports.
The user wants to know which factories have issues related to their question.
Return ONLY a numbered list of factory names or numbers that have relevant findings.
Do not include any details, descriptions, or explanations.
Do not include factories where the finding is unrelated to the question.
Sort the list by factory number in ascending order.
Do not indicate severity or prioritization in this list.
Format as a numbered list with a blank line between each item like this:
1. Factory XXX

2. Factory XXX

3. Factory XXX

Relevant findings:
{context}

Question: {question}"""
    return call_llm(prompt, max_tokens=1000)

# full report - covers all relevant factories with detailed findings
def run_workflow(question):
    if _df is None:
        return {"intake": "no data", "analysis": "no data", "report": "no data"}

    # get all relevant findings
    results = search(_df, question, top_n=80, min_score=0.35)
    context = build_context(results)

    # step 1 - intake agent extracts raw findings per factory
    intake_prompt = f"""You are a compliance data intake specialist.
Review the following audit findings and list every factory that has a finding
directly relevant to the topic asked.
For each factory write the full finding clearly.
Do not add conclusions or recommendations.
Only include factories with relevant findings.

Findings:
{context}

Topic: {question}"""
    intake_text = call_llm(intake_prompt, max_tokens=1500)

    # step 2 - analyst categorizes by severity
    analyst_prompt = f"""You are a labor compliance analyst.
Review the following factory findings and categorize each factory by severity.
Group them as Critical, Moderate, or Minor violations.
For each factory list the specific issues clearly and factually.

Findings:
{intake_text}"""
    analyst_text = call_llm(analyst_prompt, max_tokens=1500)

    # step 3 - reporter writes the full structured report
    reporter_prompt = f"""You are a compliance report writer.
Write a comprehensive factory-by-factory compliance report based on the analysis below.
Only include factories that have findings directly relevant to the topic.
Do not mention unrelated findings.

Topic: {question}

Analysis:
{analyst_text}

Format your report exactly like this:

COMPLIANCE REPORT
=================
Topic: [topic]
Total Factories with Violations: [number]
Critical Violations: [number]
Moderate Violations: [number]
Minor Violations: [number]

CRITICAL VIOLATIONS:
- Factory XXX: [detailed finding with specific numbers and dates where available]

MODERATE VIOLATIONS:
- Factory XXX: [detailed finding]

MINOR VIOLATIONS:
- Factory XXX: [detailed finding]

RECOMMENDATIONS:
[specific actionable recommendations grouped by violation type]"""
    report_text = call_llm(reporter_prompt, max_tokens=2000)

    return {
        "intake": intake_text,
        "analysis": analyst_text,
        "report": report_text
    }