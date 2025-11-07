# LLM Short Tutorials

A comprehensive collection of Databricks LLM tutorials covering fundamentals through advanced agent deployment.

## Structure

This repository is organized by topic, with each folder containing focused tutorials and examples.

### [00_setup](./00_setup/)
Workspace configuration and data preparation utilities
- Generate secrets and authentication tokens
- Create model serving endpoints
- Web scraping and dataset building

### [01_fundamentals](./01_fundamentals/)
Introduction to LLM operations on Databricks
- Basic prompting and MLflow logging
- Document parsing and chunking (LangChain & LlamaIndex)
- Vector Search index creation

### [02_rag_chains](./02_rag_chains/)
Building Retrieval-Augmented Generation chains
- Basic chat chains with LangChain
- RAG with vector search retrievers
- PyFunc and Model-as-Code patterns

### [03_agents](./03_agents/)
Intelligent agents with tools and state management
- Agents with retriever tools
- LangGraph routing and conditional logic
- Unity Catalog function integration
- Genie integration for analytics

### [04_deployment](./04_deployment/)
Deploy models to Databricks Model Serving
- Databricks Python SDK deployment
- Databricks Agents framework
- Model-as-Code deployment
- Environment configuration

### [05_applications](./05_applications/)
Interactive applications and user interfaces
- Gradio app examples
- Chat interfaces with streaming
- RAG applications
- Databricks Apps packaging

### [06_advanced_topics](./06_advanced_topics/)
Specialized topics and advanced techniques
- Synthetic data generation
- Hugging Face model operations
- Unstructured API integration
- Audio processing

## Getting Started

1. **Setup**: Start with `00_setup/` to configure your workspace
2. **Learn**: Work through `01_fundamentals/` to understand core concepts
3. **Build**: Progress through `02_rag_chains/` and `03_agents/` to build applications
4. **Deploy**: Use `04_deployment/` to serve your models
5. **Extend**: Explore `05_applications/` and `06_advanced_topics/` for advanced patterns

## Prerequisites

- Databricks workspace (AWS, Azure, or GCP)
- Unity Catalog enabled
- Model Serving endpoints provisioned
- Appropriate workspace permissions

## Shared Utilities

The `utils/` folder contains common functions used across notebooks:
- `common_functions.py` - Reusable helper functions for authentication, MLflow setup, and data formatting

## Configuration Files

The `configs/` folder contains YAML configurations for deployment:
- `rag_chain_405b.yaml` - RAG chain with Llama 405B
- `rag_chain_config.yaml` - General RAG configuration
- `databricks.yml` - Databricks Apps configuration

## Notes

- Notebooks include pip install cells for dependencies
- Prerequisites are documented in each folder's README
- Examples use Databricks-specific features (Unity Catalog, Vector Search, Model Serving)
