# Assignment 1: POS Tagging using HMM

## 1\. Introduction

Part-of-Speech (POS) tagging assigns a grammatical category (such as noun, verb, adjective, etc.) to each token in a sentence. In this assignment, we implement a POS tagger using Hidden Markov Models (HMM) with three configurations:

1. First-Order HMM: Assuming emission P(wordi | tagi) and transition P(tagi | tagi-1).
2. Second-Order HMM: Emission P(wordi | tagi) and transition P(tagi | tagi-2, tagi-1).
3. First-Order HMM with Previous-Word Emission: Emission P(wordi | tagi, wordi-1) and transition P(tagi | tagi-1).

We evaluate each configuration on the Penn Tree bank corpus, first with 36 fine-grained tags, then collapsing them into 4 broad categories (N, V, A, O). Metrics include overall accuracy, tag-wise accuracy, and various analyses through plots.

## 2\. Dataset & Preprocessing

We used a JSON-formatted POS-tagged version of the English Penn Tree bank. Each entry is a pair of:

- Sentence string (tokens separated by whitespace)
- List of corresponding POS tags

Data loading was done directly from a Google Drive URL. After splitting each sentence into tokens, we verified that the token count matched the tag count. The full corpus contained 3,914 sentences, which we randomly split into 80% train (3,131 sentences) and 20% test (783 sentences).

## 3\. Methodology

**3.1 HMM Parameter Estimation**  
We collected counts over the training set for:

- **Initial counts:** Number of sentences starting with each tag.
- **Transition counts (1st-order):** tagi-1 → tagi.
- **Transition counts (2nd-order):** (tagi-2, tagi-1) → tagi.
- **Emission counts:** tag → word.
- **Prev-word emission counts:** (tag, previous_word) → current_word.

Add-one smoothing was applied to all initial and transition probabilities to avoid zero-probability paths. Emission probabilities remained un-smoothed, directing unseen words to the most frequent tag in the entire corpus.

**3.2 Viterbi Decoding**  
Three Viterbi variants were implemented:

1. _First-Order HMM:_ δi(t) = maxu\[δi-1(u) + log P(t|u)\] + log P(wi|t).
2. _Second-Order HMM:_ δi(ti-1, ti) = maxti-2\[δi-1(ti-2, ti-1) + log P(ti|ti-2, ti-1)\] + log P(wi|ti).
3. _Prev-Word Emission:_ δi(t) = maxu\[δi-1(u) + log P(t|u)\] + log P(wi|t, wi-1).

After decoding, tags were compared to ground truth to compute:

- Overall token-level accuracy
- Tag-wise accuracy for each of the 36 tags (and similarly for the 4 collapsed tags)
- Sentence-length-based accuracy
- Confusion counts between true vs. predicted tags

Finally, we collapsed the original 36 POS tags into 4 classes—N (nouns), V (verbs), A (adjectives/adverbs), O (others)—and retrained/evaluated the first-order, second-order, and prev-word variants under this reduced tag set.

## 4\. Results

### 4.1 Overall Accuracy Comparison

The bar chart below compares the six HMM variants (36-tag vs. 4-tag, 1st-order vs. 2nd-order vs. prev-word). Values are token-level accuracy on the test set.

![Overall Accuracy Bar Chart](output1.png)

### 4.2 Tag-Wise Accuracy Heatmap (36-Tag)

This heatmap shows per-tag accuracy for two variants: first-order and prev-word emission (36-tag). Rows correspond to each of the 36 PTB tags.

![36-Tag Accuracy Heatmap](output2.png)

### 4.3 Accuracy by Collapsed Category: 36-tag vs 4-tag (1st-order)

For each of the four collapsed classes (N, V, A, O), this grouped bar chart compares the first-order accuracy under the 36-tag model versus the 4-tag model.

![Collapsed Category Bar Chart](output3.png)

### 4.4 36-Tag Confusion Matrix (Normalized, 1st-order)

Each row is a true POS tag; each column is a predicted tag. Cell values are normalized counts (fraction of true-tag occurrences predicted as each tag). Off-diagonal cells highlight systematic confusions.

![36-Tag Confusion Matrix](output4.png)

### 4.5 Accuracy vs Sentence Length (36-Tag)

This line plot shows how tagging accuracy varies with sentence length for first-order and prev-word HMM (36-tag). Each point is the average accuracy for all test sentences of that length.

![Accuracy vs Sentence Length](output5.png)

### 4.6 Tag Frequency vs Tag-Wise Accuracy (36-Tag, 1st-order)

Scatter plot with x=log(tag frequency in training) and y=tag-wise accuracy (1st-order). Rare or low-accuracy tags are annotated.

![Frequency vs Accuracy Scatter](output6.png)

### 4.7 Error Rate Reduction (36→4 tags)

For each HMM variant, we compute the reduction in error rate when collapsing from 36 tags to 4 tags: error_reduction = (err36 − err4) / err36. Higher values indicate a larger relative improvement.

![Error Rate Reduction Bar Chart](output7.png)

### 4.8 Per-Original-Tag Improvement (36→4, 1st-order)

Each bar shows (accuracy4-tag − accuracy36-tag) for each of the original 36 tags, sorted descending. Positive values mean the collapsed model outperformed the fine-grained model for that tag.

![Per-Tag Improvement Bar Chart](output8.png)

## 5\. Discussion

**36-Tag vs. 4-Tag Overall:** Collapsing from 36 tags to 4 tags produced a significant gain in overall accuracy (e.g., 1st-order: 85.61 % → 91.81 %), reducing the error rate by over 37 % (see error-rate reduction bar chart). This is because the 36-tag model's transition and emission tables are large and sparse—many tag-pairs and tag-word pairs never occur in training. By merging “NN, NNS, NNP, …” into “N,” counts become much denser and probabilities better estimated.

**First-Order vs. Second-Order:** The 2nd-order HMM yields a slight improvement over 1st-order (e.g., 85.66 % vs. 85.61 % in 36-tag), as it leverages the extra context of two preceding tags. However, the computational cost of decoding grows substantially (36³ per time step). Under 4 tags, the difference is even smaller (91.83 % vs. 91.81 %).

**Prev-Word Emission:** Conditioning emission on the previous word (rather than just the tag) performed poorly on the 36-tag model (20.59 % overall), due to extreme data sparsity in (tag, previous_word) → current_word counts. Even in the 4-tag setting, accuracy remained below 50 %, indicating that without smoothing or backoff, this approach suffers severely from unseen contexts.

**Tag-Wise Observations:** - High-frequency tags like “NN,” “DT,” “IN,” “PRP” achieve ≥90 % accuracy in the 36-tag 1st-order model. - Rare tags (e.g., “RBS,” “UH,” “WRB”) often have <60 % accuracy due to very few training examples. - The collapsed 4-tag model boosts low-frequency tags dramatically (e.g., “JJR” from 52 % → 88 %), as they now share counts with all adjectives/adverbs.

## 6\. Conclusion

We built and evaluated three HMM configurations for POS tagging on the Penn Tree bank. The best-performing setup was the 4-tag, second-order HMM (91.83 % accuracy), closely followed by the 4-tag first-order HMM (91.81 %). Collapsing from 36 fine-grained tags to 4 broad categories drastically reduced data sparsity, leading to a relative error-rate reduction exceeding 60 % in the 1st-order setting. Future work could explore smoothing techniques (for prev-word emission), CRF models, or neural-based taggers to further close the remaining error gap.

## 7\. References

- Marcus, M. P., Marcinkiewicz, M. A., & Santorini, B. (1993). Building a large annotated corpus of English: The Penn Treebank. _Computational Linguistics_, 19(2), 313-330.
- Jurafsky, D., & Martin, J. H. (2009). _Speech and Language Processing_ (2nd ed.). Pearson.
- Manning, C. D., & Schütze, H. (1999). _Foundations of Statistical Natural Language Processing_. MIT Press.
