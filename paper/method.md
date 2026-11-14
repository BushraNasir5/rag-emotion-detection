# Method

## 1. Overview

EA-RAG augments a standard retrieval-augmented generation pipeline with emotional context at three points:

1. emotion-conditioned query rewriting
2. emotion-aware retrieval ranking
3. tone-conditioned response generation

The detected user emotion guides both evidence selection and response style.

## 2. Emotion Detection

Given user query \(q\), an emotion classifier predicts emotion label \(\hat{e}\) and confidence \(p(\hat{e}\mid q)\):

\[
(\hat{e}, s_e) = f_{\text{emo}}(q)
\]

where \(\hat{e} \in \{\text{distress}, \text{neutral}, \text{positive}\}\) in this implementation.

## 3. Emotion-Conditioned Query Rewrite

The original query is rewritten to bias retrieval:

\[
q' = g_{\text{rewrite}}(q, \hat{e})
\]

For distress, the rewrite requests supportive and actionable passages; for neutral, concise factual passages; for positive, reinforcement-oriented passages.

## 4. Emotion-Aware Ranking

Retriever returns top-\(k\) candidates \(D=\{d_i\}_{i=1}^k\) with semantic similarity \(\text{sim}(q', d_i)\). We compute final score:

\[
\text{score}(d_i) =
\alpha \cdot \text{sim}(q', d_i)
+ \beta \cdot \text{emoAlign}(d_i, \hat{e})
+ \gamma \cdot \text{safety}(\hat{e})
\]

with \(\alpha+\beta+\gamma=1\), and default \((\alpha,\beta,\gamma)=(0.70,0.25,0.05)\).

## 5. Tone-Conditioned Generation

The generator consumes top-ranked documents and a tone policy:

\[
y = h_{\text{gen}}(q, D^{*}, t(\hat{e}))
\]

where tone map \(t(\cdot)\) is supportive for distress, celebratory for positive, and neutral otherwise.

## 6. Ablation Settings

- full EA-RAG
- no query rewrite
- no emotion-aware ranking
- no tone conditioning
