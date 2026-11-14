# Abstract

Emotion-aware artificial intelligence systems are essential for delivering personalized and contextually appropriate user interactions. Traditional retrieval-augmented generation systems prioritize factual grounding but frequently ignore user affect, producing outputs that are correct yet emotionally misaligned. We propose Emotion-Aware Retrieval-Augmented Generation (EA-RAG), a modular framework that injects emotion signals into query rewriting, retrieval ranking, and response generation. The system combines a transformer-based emotion detector with an emotion-aware ranking function that reweights semantic similarity by emotional alignment and safety priority. We further introduce tone-conditioned prompting to align response style with user emotion while preserving groundedness. Experiments using a reproducible benchmark pipeline show measurable gains in top-rank retrieval sensitivity for emotion-aware ranking and provide a practical foundation for evaluating emotional alignment in grounded dialogue systems. These results support integrating affective computing into RAG as a necessary step toward reliable and human-centric assistants.

# Introduction

Large language models have substantially improved open-domain conversational quality. When paired with retrieval-augmented generation (RAG), they can ground answers in external evidence and reduce hallucination risk. However, most RAG pipelines optimize only for semantic relevance and factual support. In practice, users communicate through both information and emotion; a system that answers factually but with an inappropriate tone can still fail the interaction objective.

This gap is particularly visible in scenarios such as student support, customer assistance, and wellbeing-oriented conversations. For example, a user expressing overwhelm requires both accurate content and emotionally appropriate phrasing. Existing RAG systems typically treat such inputs as standard factual queries, ignoring emotional cues during retrieval and generation.

To address this limitation, we present Emotion-Aware Retrieval-Augmented Generation (EA-RAG), a framework that makes emotion a first-class signal in the RAG loop. EA-RAG detects user emotion from the input query, rewrites the query to bias retrieval toward contextually suitable evidence, reranks retrieved passages using an emotion-aware scoring function, and conditions generation on an explicit tone policy.

The contributions of this project are:

1. A practical, reproducible EA-RAG architecture that integrates emotion detection into both retrieval and generation stages.
2. An emotion-aware ranking formulation that combines semantic relevance, emotional alignment, and safety-oriented priors.
3. A complete baseline and ablation workflow (vanilla RAG, prompt-only, and module-removal studies) with quantitative outputs and paper-ready assets.

Overall, this work demonstrates an implementation-focused pathway for building emotionally aligned, knowledge-grounded assistants and provides an extensible foundation for stronger future evaluations on larger public benchmarks.
