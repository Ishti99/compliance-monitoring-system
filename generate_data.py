import boto3
import json
import random
import os

os.environ["AWS_PROFILE"] = "GSB570-BedrockOnly-490332585640"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

def get_client():
    session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
    return session.client("bedrock-runtime", region_name="us-west-2")

def call_llm(prompt):
    client = get_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.9,
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

compliance_areas = [
    "Working Hours & Overtime",
    "Wages & Benefits",
    "Occupational Safety & Health",
    "Child Labor",
    "Forced Labor",
    "Discrimination",
    "Freedom of Association & Collective Bargaining",
    "Contracts & Human Resources",
    "Harassment & Abuse",
    "Emergency Preparedness"
]

# each factory gets a random subset of compliance areas
# some factories will have many issues, some very few
def get_areas_for_factory(factory_num):
    # randomly decide how many areas to include
    # some factories are mostly compliant, some have many issues
    r = random.random()
    if r < 0.15:
        # mostly compliant factory - only 1 or 2 areas with findings
        num_areas = random.randint(1, 2)
    elif r < 0.4:
        # few issues - 3 to 4 areas
        num_areas = random.randint(3, 4)
    elif r < 0.7:
        # moderate issues - 5 to 6 areas
        num_areas = random.randint(5, 6)
    else:
        # many issues - 7 to 10 areas
        num_areas = random.randint(7, 10)
    return random.sample(compliance_areas, num_areas)

def generate_finding(factory_name, area, factory_profile):
    prompt = f"""You are writing a realistic garment factory audit report for {factory_name}.

Factory profile: {factory_profile}

Write a realistic audit finding for the compliance area: {area}

Important instructions:
- Write like a real auditor wrote this - include some background context, inspector observations, worker interview notes, and dates where appropriate
- Sometimes include reference numbers or audit codes
- The finding can be a violation (Critical, Moderate, or Minor severity) OR compliant
- Make it verbose and realistic - real audit reports are not always to the point
- Mix in some bureaucratic language and follow-up notes
- Do NOT use bullet points - write in paragraphs like a real report
- Length should be 3-5 sentences minimum
- Sometimes mention specific numbers of workers interviewed, specific dates, specific measurements
- Severity level (if violation): state it clearly at the end as "Severity: Critical", "Severity: Moderate", "Severity: Minor", or "Status: Compliant"
- Do not start with the factory name

Just write the finding paragraph, nothing else."""

    return call_llm(prompt)

def get_factory_profile():
    profiles = [
        "large export-oriented factory with 800+ workers, multiple product lines, experienced management team",
        "small family-owned factory with 50-100 workers, limited HR capacity, first time being audited",
        "medium-sized factory with 200-400 workers, recently changed ownership, transitioning management",
        "factory with history of compliance issues, under corrective action plan from previous audit",
        "newly established factory with 100-200 workers, young workforce, still building compliance systems",
        "well-established factory with strong compliance history, ISO certified, dedicated compliance officer",
        "subcontractor factory working for multiple brands, high production pressure, seasonal workforce",
        "factory in rural area with limited access to legal support, mostly migrant workers",
        "factory with active union, good worker-management relations, strong grievance mechanism",
        "factory recovering from recent fire incident, under special monitoring by local authorities"
    ]
    return random.choice(profiles)

# generate data for 100 factories
output_path = "data/compliance_data.txt"
os.makedirs("data", exist_ok=True)

print("starting data generation for 100 factories...")
print("this will take a while - each factory needs multiple LLM calls\n")

with open(output_path, "w", encoding="utf-8") as f:
    for i in range(1, 101):
        factory_name = f"Factory {i:03d}"
        profile = get_factory_profile()
        areas = get_areas_for_factory(i)

        print(f"generating {factory_name} ({len(areas)} compliance areas)...")

        f.write(f"{'='*60}\n")
        f.write(f"{factory_name} - Audit Report\n")
        f.write(f"Factory Profile: {profile}\n")
        f.write(f"{'='*60}\n\n")

        for area in areas:
            print(f"  - {area}")
            finding = generate_finding(factory_name, area, profile)
            f.write(f"[{area}]\n")
            f.write(finding.strip())
            f.write("\n\n")

        f.write("\n")

print(f"\ndone! data saved to {output_path}")
print("you can now load this file in the Streamlit app")