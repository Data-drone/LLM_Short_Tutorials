# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building Test Datasets
# MAGIC
# MAGIC Let's hit arxiv to build a midsized dataset
# MAGIC
# MAGIC *NOTE* This uses scholarly which uses Google Scholar which has API access limits
# MAGIC So don't use this to scrape all papers

# COMMAND ----------

# MAGIC %pip install arxiv langchain==0.1.16 pypdf
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = 'brian_gen_ai'
schema = 'larger_datasets'
arxiv_volume = 'arxiv'

vol_path = f"/Volumes/{catalog}/{schema}/{arxiv_volume}"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{arxiv_volume}")

# COMMAND ----------

# import lib and start exploring
import arxiv
import os

# Construct the default API client.
client = arxiv.Client()

# COMMAND ----------

# MAGIC %md # Retrieve some papers

# COMMAND ----------

# Search for the paper with ID "2312.10997"
rag_paper = '2312.10997'
RAG_summary_paper = arxiv.Search(id_list=[rag_paper])

# view search results
for r in client.results(RAG_summary_paper):
    first_result = r
    first_title = r.title

print(first_result.title)

# COMMAND ----------

# MAGIC %md We didn't find any easy to use citation search python tools \
# MAGIC so we will have to find other ways \
# MAGIC We can parse a paper to find it's citations with basic regex for arxiv

# COMMAND ----------

# check exists
## Note this doesn't check for file corruption
def download_paper(arxiv_code):
    if not os.path.exists(os.path.join(vol_path, f"{arxiv_code}.pdf")):
        paper_search = arxiv.Search(id_list=[arxiv_code])
        for r in client.results(paper_search):
            split_parts = r.entry_id.rsplit('/', 1)
            arxiv_string = split_parts[-1]
            assert len(arxiv_string) > 1

            print(f'Downloading {arxiv_string}')
            r.download_pdf(dirpath=vol_path, filename=f"{arxiv_string}.pdf")

            return f"{arxiv_string}.pdf"

    else:
        print(f'Paper {arxiv_code} already downloaded')   
        return f"{arxiv_code}.pdf"


download_paper(rag_paper)

os.listdir(vol_path)

# COMMAND ----------

from langchain_community.document_loaders import PyPDFLoader
import re

# parse and extract
def parse_and_extract(full_path):

    paper_number = full_path.rsplit('/', 1)
    arxiv_string = paper_number[-1].replace(".pdf", "")
    print(f'processing {arxiv_string}')

    loader = PyPDFLoader(full_path)
    doc = loader.load()

    arxiv_pattern = r"(?:[\w\s]*?)arXiv:(\d+\.\d+\d*(?:v\d+)?)\b"
    
    matches = []
    for page in doc:
        match = re.findall(arxiv_pattern, page.page_content, re.IGNORECASE)
        if len(match) > 0:
            matches.extend(match)

    # make sure it's not circular
    filter_matches=[x for x in matches if x[0:10]!=arxiv_string[0:10]]

    return filter_matches


# COMMAND ----------

import random
import time

rag_arxiv_references = parse_and_extract(os.path.join(vol_path, f"{rag_paper}.pdf"))

for paper in rag_arxiv_references: 

    wait_time = random.uniform(1, 5)
    time.sleep(wait_time)

    filename = download_paper(paper)
    further_references = parse_and_extract(os.path.join(vol_path, filename))
    
    for nested_paper in further_references:
        
        wait_time = random.uniform(1, 5)
        time.sleep(wait_time)

        download_paper(nested_paper)

os.listdir(vol_path)


