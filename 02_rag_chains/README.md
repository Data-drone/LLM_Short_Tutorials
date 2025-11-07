# 02 RAG Chains

## Overview
Building and logging Retrieval-Augmented Generation (RAG) chains using both PyFunc and Model-as-Code approaches.

## Learning Objectives
- Build basic chat chains with LangChain
- Create RAG chains with vector search retrievers
- Log models using MLflow PyFunc and native LangChain integration
- Implement model-as-code pattern with `mlflow.models.set_model()`

## Prerequisites
- Vector Search index created (from `01_fundamentals/01_parsing_chunking_langchain.py`)
- Databricks serving endpoints: LLM model and embedding model

## Notebooks
- `00_basic_chain_logging.py` - Basic chains with traditional MLflow logging
- `01_modelfile_basic.py` - Model-as-code approach for basic chains
- `02_modelfile_advanced.py` - Advanced chain patterns with model-as-code

