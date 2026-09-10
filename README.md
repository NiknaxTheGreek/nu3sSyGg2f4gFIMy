# Potential Talents - Candidate Ranking with NLPL Model 40 Word2Vec and Rule-Based Relevance

## Executive Summary

This project develops a candidate-ranking workflow that combines **transparent rule-based relevance screening, semantic ranking with NLPL Model 40 Word2Vec, and recruiter-driven reranking**. The analysis is performed on **52 unique job titles** derived from 104 candidate records so duplicate titles do not receive extra weight in evaluation. Using the pretrained **English CoNLL17 Word2Vec 100-dimensional model (NLPL model 40)**, the Stage-1 ranking achieved **NDCG@10 = 0.948** for `aspiring human resources` and **0.935** for `seeking human resources`, with a mean of **0.942**.

The model covered **100% of the 374 meaningful token occurrences** used across the 52 titles and two recruiter queries, with **zero unique out-of-vocabulary tokens** after the audited preprocessing rules. Recruiter feedback was then tested by shifting the query vector toward a starred candidate. At a conservative **30% feedback influence**, the two demonstration candidates both moved from **rank 2 to rank 1** while retaining **9/10** and **10/10** of the original top-ten shortlist respectively.

## Project Objectives

**Overall objective:** Develop and evaluate a transparent, reproducible candidate-ranking workflow that prioritizes Human Resources candidates for the two recruiter searches using NLPL Model 40 Word2Vec, rule-based relevance, and recruiter feedback.

The project is guided by six sub-objectives:

1. **Audit and prepare the candidate data** by identifying duplicate job titles, reducing the 104 source records to 52 unique titles for unbiased evaluation, and characterizing the title vocabulary.
2. **Define an auditable relevance benchmark** using occupational relevance and search-intent alignment through the rule score $R = H(0.70 + 0.30I)$.
3. **Validate the reference procedure independently** by comparing its ordinal relevance levels with the manually assigned 0-3 human relevance grades.
4. **Rank candidates semantically with NLPL Model 40** by representing titles and recruiter queries with mean Word2Vec embeddings and ordering candidates by cosine similarity.
5. **Evaluate ranking quality and coverage** using NDCG@10, top-ranked candidate inspection, and vocabulary/OOV checks for both recruiter queries.
6. **Evaluate recruiter-driven reranking** by shifting the query toward starred candidates, measuring the personalization-stability trade-off across 34 strong scenarios, and translating the findings into practical recommendations and limitations.

## Problem Definition

Finding the right candidate is a ranking problem rather than a simple search problem. A recruiter already has a sourced pool of potential candidates and must decide which profiles deserve attention first. The supplied dataset contains **104 anonymized candidate records** with `id`, `job_title`, location, connection count, and an initially empty `fit` field. This project ranks candidates using job-title evidence only, then shows how the ranking can adapt when a recruiter stars an ideal candidate.

The two recruiter searches are `aspiring human resources` and `seeking human resources`.

## Data

Because both ranking approaches use only `job_title`, the analysis is performed once per **unique title**. The source data contain 104 rows but only **52 unique job titles**. Fourteen titles repeat, and the most repeated title frequency is **7**. Deduplication prevents repeated titles from artificially dominating frequency analysis or ranking metrics while preserving the ability to map title-level scores back to all original candidate IDs.

| Dataset characteristic | Value |
|---|---:|
| Source candidate records | 104 |
| Unique job titles evaluated | 52 |
| Repeated unique titles | 14 |
| Maximum exact-title frequency | 7 |
| Titles containing `Human Resources` | 31 (59.6%) |
| Titles with direct HR evidence | 34 (65.4%) |
| Titles containing `aspiring` | 12 |
| Titles containing `seeking` | 10 |

### Dataset Structure and Vocabulary

The title vocabulary contains frequent explicit HR signals but also sparse related terminology. This supports the combined design: rules provide an auditable reference for clear role and intent evidence, while Word2Vec supplies semantic similarity when wording varies. The title audit removes standalone years and phone numbers, treats punctuation and hyphens as separators, removes the structural connector `at`, and drops isolated uppercase initials created by punctuation. Token lookup is exact-case first and lowercase second; possessives are reduced only when their base form exists in the model vocabulary. No synonym substitution is added.

For Model 40, the 52 titles contain **368 meaningful token occurrences** and the two queries add six more, for **374 total**. All 374 resolve to vectors. Across the analysis there are **208 unique meaningful tokens** and **0 unique OOV tokens**.

## Ranking Study

### 1. NLPL Model 40 Word2Vec Semantic Ranking

The embedding model is **NLPL model 40: English CoNLL17 Word2Vec, 100 dimensions, continuous skip-gram**, with a vocabulary header of **4,027,169 words**. The official archive is approximately **3.03 GB** and is not bundled with the repository. Each title and recruiter query is represented by the mean of its available word vectors, and candidates are ranked by cosine similarity to the query representation.

Official model archive: <https://vectors.nlpl.eu/repository/20/40.zip>

### 2. Rule-Based Relevance Reference

A transparent relevance score provides an independent query-specific benchmark:

$$R = H(0.70 + 0.30I)$$

where $H$ represents occupational relevance to Human Resources and $I$ represents alignment with the search intent (`aspiring` or `seeking`). Direct HR evidence receives the strongest occupational relevance; adjacent functions such as staffing, recruiting, talent management, benefits, and compensation receive partial relevance. Clear employer solicitations are assigned zero candidate relevance.

## Stage-1 Results

**NDCG@10** measures how well the semantic top-ten ordering places the most relevant titles near the top according to the graded rule-based benchmark.

| Recruiter query | NDCG@10 |
|---|---:|
| Aspiring Human Resources | **0.948** |
| Seeking Human Resources | **0.935** |
| Mean | **0.942** |

For `aspiring human resources`, the first three titles are **Aspiring Human Resources Specialist (ID 6)**, **Aspiring Human Resources Professional (ID 3)**, and **Aspiring Human Resources Manager, seeking internship in Human Resources (ID 73)**. For `seeking human resources`, the first three are **Seeking Human Resources Opportunities (ID 28)**, **Seeking Human Resources Position (ID 99)**, and **Aspiring Human Resources Manager, seeking internship in Human Resources (ID 73)**.

## Independent Human Relevance Assessment

The 52 unique titles were independently assigned 0-3 human relevance grades using title information only. These labels were not used to train Model 40 or construct the rule score.

| Human vs. rule-based agreement | Result |
|---|---:|
| Exact ordinal agreement | **39/52 - 75.0%** |
| Quadratic weighted Cohen's kappa | **0.888** |
| Spearman correlation | **0.871** |

## Recruiter Feedback and Dynamic Reranking

When a recruiter stars a candidate, the query vector is shifted toward that candidate's normalized title vector:

$$q_2 = \operatorname{normalize}((1-w)q + wd_\star)$$

The feedback sweep evaluates **34 strong title/query scenarios** (`R >= 0.85`) at 10%, 20%, 30%, 40%, and 50% feedback influence.

| Feedback weight | Median starred-title rank | Median original top-10 retained |
|---:|---:|---:|
| 10% | 8.0 | 9/10 |
| 20% | 6.5 | 9/10 |
| 30% | 5.0 | 9/10 |
| 40% | 3.5 | 8/10 |
| 50% | 1.0 | 8/10 |

The sweep shows a personalization-stability trade-off rather than a single universally optimal weight. A **30% setting** is retained as a conservative demonstration point because it preserves a median **9/10** of the original top ten and performs strongly in the two representative examples, but it is **not claimed as a global optimum**.

| Query | Starred candidate | Rank change | Top-10 retained | NDCG@10 |
|---|---|---:|---:|---:|
| Aspiring HR | ID 3 - *Aspiring Human Resources Professional* | **2 -> 1** | 9/10 | 0.948 -> **0.955** |
| Seeking HR | ID 99 - *Seeking Human Resources Position* | **2 -> 1** | 10/10 | 0.935 -> **0.940** |

## Practical Implications

Use the rule system to flag profiles with no defensible role relevance, Model 40 to rank the remaining titles semantically, and recruiter feedback as a controlled personalization layer. A universal cosine-similarity cutoff is not recommended because absolute similarity depends on the query and candidate pool. Recruiter oversight remains important because pretrained embeddings may reflect patterns in their training corpus and recruiter selections can propagate human preferences into later rankings.

## Limitations

- Mean-pooled Word2Vec does not fully model sentence directionality or context.
- Equal averaging can dilute important terms in long job titles.
- Only `job_title` is used; skills and experience absent from the title cannot affect ranking.
- The rule framework must be redefined and validated for occupations outside Human Resources.
- Feedback weight is a product choice with a personalization-stability trade-off; the current experiment does not establish a universal optimum.

## Project Structure

```text
project/
|-- README.md
|-- potential-talents.csv
|-- unique-job-titles-labelled.csv
|-- 01_data_rule_based_ranking.ipynb
|-- 02_stage1_word2vec.ipynb
|-- 03_recruiter_feedback.ipynb
|-- Technical_Report.pdf
|-- Business_Recommendations.pdf
|-- requirements.txt
`-- .gitignore
```

## Reproducibility

Notebook 01 recomputes the dataset audit, duplicate analysis, rule scores, and independent human-vs-rule validation from the two included CSV files. Notebooks 02 and 03 use **NLPL Model 40 only**. Place the official `40.zip` archive in the project root (or set `NLPL_MODEL40_ZIP` to its path). The notebooks stream `model.txt` directly from the archive and extract only the vectors needed by the 52 titles and two queries, avoiding the memory cost of loading the complete 4-million-word model into RAM.
