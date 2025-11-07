# 05 Applications

## Overview
Building interactive applications with Gradio and deploying them on Databricks driver-proxy or as Databricks Apps.

## Learning Objectives
- Launch Gradio apps on Databricks clusters
- Build chat interfaces with streaming responses
- Create RAG applications with document retrieval
- Package and deploy Databricks Apps with YAML configs

## Prerequisites
- Deployed model endpoints (from `04_deployment/`)
- Databricks secrets configured for authentication

## Notebooks & Apps
- `00_building_apps.py` - Notebook launcher for various app examples
- `basic_test_app.py` - Simple Gradio test application
- `chat_app.py` - Basic chat interface
- `chat_to_docs_app.py` - RAG chat with document retrieval
- `adv_chat_to_docs_app.py` - Advanced RAG with better UX
- `expansion_and_rerank_app.py` - Enhanced RAG with query expansion and reranking
- `adv_agent_app/` - Packaged agent app with Databricks Apps config
- `magic_8_ball/` - Example Databricks App with streaming

