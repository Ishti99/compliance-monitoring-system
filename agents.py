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

def run_single_agent(question):
    if _df is None:
        return "no data loaded"
    results = search(_df, question, top_n=80, min_score=0.25)
    context = build_context(results)
    prompt = f"""You are a compliance analyst reviewing garment factory audit reports.
The user wants to know which factories have issues related to their question.
Return ONLY a numbered list of factory names or numbers that have relevant findings.
Do not include any details, descriptions, or explanations.
Do not include factories where the finding is unrelated to the question.
Just the factory name or number, one per line, nothing else.
Sort the list by factory number in ascending order.
Format as a numbered list like this:
1. Factory XXX

2. Factory XXX

Relevant findings:
{context}

Question: {question}"""
    return call_llm(prompt, max_tokens=1000)

def run_workflow(question):
    if _df is None:
        return {"intake": "no data", "analysis": "no data", "report": "no data"}

    results = search(_df, question, top_n=80, min_score=0.25)
    batches = [results[i:i+15] for i in range(0, len(results), 15)]

    all_intake = []
    batch_summaries = []

    for batch in batches:
        batch_context = build_context(batch)

        intake_prompt = f"""You are a compliance data intake specialist.
Review the following audit findings and list every factory that has a finding
directly relevant to the topic asked.
For each factory write the full finding clearly.
Do not add conclusions or recommendations.
Only include factories with relevant findings.

Findings:
{batch_context}

Topic: {question}"""
        intake_text = call_llm(intake_prompt, max_tokens=1500)
        all_intake.append(intake_text)

        # get a short one-liner per factory so the reporter doesn't choke
        analyst_prompt = f"""You are a labor compliance analyst.
Review the following factory findings and for each factory produce a ONE LINE summary.
Format exactly like this:
- Factory XXX: [severity] - [one sentence describing the main violation]

Only include factories with actual violations. Be very concise.

Findings:
{intake_text}"""
        summary = call_llm(analyst_prompt, max_tokens=800)
        batch_summaries.append(summary)

    combined_intake = "\n\n".join(all_intake)
    combined_summaries = "\n".join(batch_summaries)

    reporter_prompt = f"""You are a compliance report writer.
Based on the factory summaries below, write a comprehensive compliance report.
Group factories by severity - Critical, Moderate, or Minor.
Include ALL factories listed in the summaries.
Do not drop any factory from the report.

Topic: {question}

Factory summaries:
{combined_summaries}

Format your report exactly like this:

COMPLIANCE REPORT
=================
Topic: [topic]
Total Factories with Violations: [number]
Critical Violations: [number]
Moderate Violations: [number]
Minor Violations: [number]

CRITICAL VIOLATIONS:
- Factory XXX: [detailed finding]

MODERATE VIOLATIONS:
- Factory XXX: [detailed finding]

MINOR VIOLATIONS:
- Factory XXX: [detailed finding]

RECOMMENDATIONS:
[specific actionable recommendations grouped by violation type]"""
    report_text = call_llm(reporter_prompt, max_tokens=4000)

    return {
        "intake": combined_intake,
        "analysis": combined_summaries,
        "report": report_text
    }