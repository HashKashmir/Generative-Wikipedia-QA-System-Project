# Generative Wikipedia QA System

This project implements a small-scale **generative question-answering (QA)** system that answers user queries about machine learning topics using crawled content from Wikipedia. The system combines **semantic search** with **generative natural language processing** (NLP) to return rich, coherent answers drawn from multiple relevant sources.

## Features

- **Web Crawler**: Scrapes and cleans over 2,500 Wikipedia pages on machine learning using `BeautifulSoup`
- **Semantic Search Engine**: Embeds text with `SentenceTransformers` and builds a fast vector index using `FAISS`
- **Generative QA**: Uses a `Flan-T5` transformer model to generate human-like answers based on retrieved chunks
- **Query Expansion & Feedback**: Enhances retrieval with synonym-based expansion and relevance scoring
- **End-to-End Pipeline**: From crawling to inference, fully modular and easy to adapt to new domains

## Technologies Used

- Python
- Hugging Face Transformers (`Flan-T5`)
- SentenceTransformers
- FAISS (Facebook AI Similarity Search)
- BeautifulSoup
- scikit-learn
- Google Colab (for development)

## How It Works

1. **Crawl Wikipedia Pages**: Select relevant articles using keyword filters
2. **Preprocess & Embed**: Clean, tokenize, and convert paragraphs into dense vector embeddings
3. **Semantic Retrieval**: Use cosine similarity via FAISS to retrieve top-k relevant chunks
4. **Generate Answers**: Feed the retrieved context + question to a generative model (Flan-T5)
5. **Return Answer**: Output a full-sentence, context-aware response

## Notebook Sections

- `# Crawler`: Wikipedia scraping and text extraction
- `# Preprocess and Embed`: Sentence segmentation, cleaning, and embedding
- `# Question and Answering`: Retrieval and generation logic
- `# Utility Functions`: Query expansion, reranking, and feedback loop

## Example

```text
Q: What is the difference between supervised and unsupervised learning?

A: Supervised learning uses labeled data to train models, while unsupervised learning discovers patterns from unlabeled data. Supervised methods include classification and regression, whereas unsupervised methods include clustering and dimensionality reduction.
