# Labor Compliance Monitoring System

This is my final project for GSB 570 - Generative AI for 
Business Analytics at Cal Poly SLO (Spring 2026). The idea 
came from my background in international development and my 
wife's work at International Labour Organization (ILO), which does labor 
compliance audits in garment factories. I wanted to build 
something that could actually be useful in that space.

## The Problem

Factory compliance audits generate a lot of text. An 
auditor visits a factory, interviews workers, reviews 
records, and writes a detailed report. When you have 
hundreds of factories this becomes unmanageable manually. 
A compliance officer shouldn't have to read through 
thousands of pages to find which factories have child 
labor violations or overtime issues. That's what this 
tool tries to solve.

## What it does

You load audit data, ask a question in plain English, 
and the system finds the relevant factories and tells 
you what the issues are. For example:

- "Which factories had overtime violations?"
- "Are there any child labor issues?"
- "What safety problems were found?"

You get either a quick list of affected factories or a 
full compliance report with severity levels (Critical, 
Moderate, Minor), detailed findings, and recommendations. 
You can also download the report.

## How it works under the hood

1. **ETL** - reads audit files (txt or pdf) and extracts 
   the text. If it's a pdf, pdfplumber handles the 
   extraction. An LLM then structures the raw text.

2. **Chunking** - splits the text by factory and 
   compliance area so each chunk stays focused on 
   one factory's finding for one topic.

3. **Embeddings** - each chunk gets converted into a 
   1024-dimensional vector using Amazon Titan. These 
   vectors capture the semantic meaning of the text.

4. **RAG** - when you ask a question, it gets embedded 
   too and compared against all stored chunks using 
   cosine similarity. The most relevant chunks come back.

5. **Agents** - a 3-agent workflow processes the results. 
   Agent 1 collects raw findings. Agent 2 analyzes and 
   categorizes by severity. Agent 3 writes the report.

6. **Streamlit** - ties everything together in a simple 
   web interface anyone can use.

## Tech stack

- Python
- Amazon Bedrock (Titan Embeddings v2, DeepSeek v3)
- Strands Agents SDK
- Streamlit
- spaCy
- NumPy, Pandas, Matplotlib

## How to run it

### What you need
- Python 3.10+
- AWS account with Bedrock access
- AWS CLI configured with your profile

### Install dependencies

```bash
git clone https://github.com/yourusername/compliance-monitoring
cd compliance-monitoring
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Update your AWS profile

In these files replace `GSB570-BedrockOnly-490332585640` 
with your own AWS profile name:
- `app.py`
- `agents.py`
- `embeddings.py`
- `etl.py`

### Run the app

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How to use it

1. Click **Load Default Compliance Data** in the sidebar
2. Or upload your own audit file (txt or pdf)
3. Type a compliance question
4. **Quick Answer** - fast list of affected factories
5. **Full Report** - detailed analysis with severity 
   chart and downloadable report

Note: the first load takes around 45 minutes to embed 
all 645 chunks. After that it uses a cached file and 
loads instantly.

## Files

```
compliance_system/
├── app.py              # main Streamlit app
├── agents.py           # single agent + 3-agent workflow
├── embeddings.py       # chunking and embeddings
├── rag.py              # retrieval logic
├── etl.py              # file reading and extraction
├── generate_data.py    # script used to generate the dataset
├── requirements.txt
├── data/
│   └── compliance_data.txt
└── README.md
```

## The dataset

I generated synthetic audit data for 100 factories using 
an LLM. Each factory has findings across up to 10 
compliance areas - Working Hours, Wages, Safety, Child 
Labor, Forced Labor, Discrimination, Freedom of 
Association, Contracts, Harassment and Emergency 
Preparedness. The data was intentionally made verbose 
and realistic to test how well the RAG system handles 
messy real-world audit language.

## Important

Do not commit your AWS credentials or any .env files. 
The .gitignore already excludes the pkl cache files 
since they are large and can be regenerated.

## Author

Md. Ishtiaque Alam
MSBA, Cal Poly SLO - Spring 2026
