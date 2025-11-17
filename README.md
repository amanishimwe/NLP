# NLP Tutorial on Word to Vec Embedding

## Overview

This repository contains a comprehensive implementation of the **Continuous Bag of Words (CBOW)** model, one of the two main architectures in Word2Vec. The CBOW model learns dense vector representations (embeddings) of words by predicting a target word from its surrounding context words. These embeddings capture semantic and syntactic relationships between words, making them valuable for various natural language processing tasks.

## What is Word2Vec?

Word2Vec is a technique for learning word embeddings from large text corpora. It was introduced by researchers at Google in 2013. The key idea is that words appearing in similar contexts should have similar vector representations. Word2Vec offers two architectures:

1. **CBOW (Continuous Bag of Words)**: Predicts a target word from surrounding context words
2. **Skip-gram**: Predicts surrounding context words from a target word

This tutorial focuses on the CBOW architecture.

## How CBOW Works

### The Core Principle

CBOW operates on the principle: **"You shall know a word by the company it keeps"** (Firth, 1957). Given a sequence of words, CBOW:

1. Takes a window of context words (e.g., 2 words before and 2 words after)
2. Averages their embeddings
3. Predicts the target word at the center of the window

### Example

For the sentence: "The quick brown fox jumps"

With a window size of 2, the model creates training pairs:
- Context: ["The", "quick", "jumps", "over"] → Target: "brown"
- Context: ["quick", "brown", "over", "the"] → Target: "fox"

## Code Walkthrough

### 1. Installation and Setup

```python
!pip3 install -U gensim matplotlib scikit-learn torch
```
**Purpose**: Installs required libraries:
- **gensim**: For word embedding utilities (optional, for comparison)
- **matplotlib**: For visualization
- **scikit-learn**: For t-SNE dimensionality reduction
- **torch**: PyTorch deep learning framework

### 2. Data Download
The code downloads the **text8 corpus**, a cleaned dataset of Wikipedia articles containing approximately 17 million words. This corpus is preprocessed to remove special characters and formatting, making it ideal for word embedding training.

**Key Features**:
- Checks if data already exists to avoid re-downloading
- Uses `curl` for macOS compatibility
- Extracts the corpus automatically

### 3. Data Preprocessing

#### Loading the Corpus
```python
words = text.split()
```
Splits the text into individual words, creating a list of ~17 million tokens.

#### Vocabulary Building
```python
min_count = 5
vocab = {word: count for word, count in word_counts.items() if count >= min_count}
```

**Purpose**: 
- Counts word frequencies
- Filters out rare words (appearing less than `min_count` times)
- Creates word-to-index and index-to-word mappings

**Why filter rare words?**
- Reduces vocabulary size and computational complexity
- Rare words often lack sufficient context for meaningful embeddings
- Improves model stability and generalization

### 4. Training Data Preparation

#### Creating Context-Target Pairs

```python
def create_cbow_data(words_idx, window_size):
    for i in range(window_size, len(words_idx) - window_size):
        context = words_idx[i-window_size:i] + words_idx[i+1:i+window_size+1]
        target = words_idx[i]
        context_target_pairs.append((context, target))
```

**How it works**:
- Slides a window of size `2*window_size` across the text
- For each position, extracts context words (before and after)
- Creates (context, target) pairs for training

**Example with window_size=2**:
- Text: `[w1, w2, w3, w4, w5, w6]`
- At position 3: context = `[w1, w2, w4, w5]`, target = `w3`

### 5. CBOW Model Architecture

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context)  # Lookup embeddings
        embeds = embeds.mean(dim=1)        # Average pooling
        out = self.linear(embeds)          # Project to vocab size
        return out
```

#### Architecture Components

1. **Embedding Layer** (`nn.Embedding`):
   - Maps word indices to dense vectors
   - Learnable parameters: `vocab_size × embedding_dim`
   - Each word gets a unique vector representation

2. **Average Pooling**:
   - Averages all context word embeddings
   - Creates a single vector representing the context
   - This is the "bag of words" aspect (order doesn't matter)

3. **Linear Layer**:
   - Projects the averaged embedding to vocabulary size
   - Outputs logits for each word in vocabulary
   - Used to predict the target word

#### Forward Pass Flow

```
Context words: [w1, w2, w4, w5]
    ↓
Embedding lookup: [emb1, emb2, emb4, emb5]  (each: embedding_dim)
    ↓
Average pooling: (emb1 + emb2 + emb4 + emb5) / 4  (embedding_dim)
    ↓
Linear projection: logits  (vocab_size)
    ↓
Softmax: probabilities over vocabulary
```

### 6. Training Process

#### Loss Function
```python
criterion = nn.CrossEntropyLoss()
```

**Cross-Entropy Loss**: Measures how well the model predicts the target word. The model learns by minimizing this loss.

#### Optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam Optimizer**: Adaptive learning rate optimizer that combines advantages of AdaGrad and RMSProp.

#### Training Loop

For each epoch:
1. **Forward Pass**: Context words → Model → Predicted word probabilities
2. **Loss Calculation**: Compare predictions with actual target word
3. **Backward Pass**: Compute gradients using backpropagation
4. **Parameter Update**: Adjust model weights to reduce loss

**Key Training Details**:
- **Batch Processing**: Processes multiple examples simultaneously for efficiency
- **Shuffling**: Randomizes data order each epoch to improve learning
- **Progress Tracking**: Monitors loss to ensure model is learning

### 7. Extracting Word Embeddings

After training, the learned embeddings are extracted from the embedding layer:

```python
embeddings = model.embeddings.weight.data.cpu().numpy()
```

**Result**: A matrix of shape `(vocab_size, embedding_dim)` where each row is a word's embedding vector.

### 8. Visualization

#### Training Loss Plot
Shows how the model's loss decreases over epochs, indicating learning progress.

#### t-SNE Visualization
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Reduces high-dimensional embeddings to 2D for visualization
- Words with similar meanings cluster together in the 2D space
- Helps understand what relationships the model has learned

### 9. Finding Similar Words

```python
def find_similar_words(word, embeddings, word_to_idx, idx_to_word, top_k=10):
    # Calculate cosine similarity
    similarities = np.dot(normalized_embeddings, word_embedding_norm)
    # Return top k most similar words
```

**Cosine Similarity**: Measures the cosine of the angle between two vectors. Words with similar meanings have embeddings pointing in similar directions, resulting in high cosine similarity.

**Use Cases**:
- Finding synonyms
- Discovering semantic relationships
- Understanding word associations

## Key Concepts Explained

### Why Average Pooling?

CBOW uses average pooling (mean) instead of concatenation because:
- **Order Independence**: CBOW treats context as a "bag" - word order doesn't matter
- **Fixed Size**: Produces a single vector regardless of context size
- **Efficiency**: Simpler and faster than attention mechanisms

### Embedding Dimensions

- **embedding_dim = 100**: Each word is represented by a 100-dimensional vector
- Higher dimensions can capture more nuanced relationships but require more data and computation
- Lower dimensions are faster but may lose information

### Window Size

- **window_size = 2**: Uses 2 words before and 2 words after (total 4 context words)
- Larger windows capture more context but may include irrelevant words
- Smaller windows focus on immediate context

## Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 100 | Dimension of word embeddings |
| `window_size` | 2 | Number of words before/after target |
| `min_count` | 5 | Minimum word frequency to include |
| `learning_rate` | 0.001 | Step size for optimization |
| `batch_size` | 128 | Number of examples per batch |
| `num_epochs` | 5 | Number of training iterations |

## Applications of Word Embeddings

1. **Semantic Similarity**: Finding words with similar meanings
2. **Document Classification**: Using embeddings as features
3. **Machine Translation**: Cross-lingual embeddings
4. **Question Answering**: Understanding query semantics
5. **Sentiment Analysis**: Capturing emotional context
6. **Recommendation Systems**: Understanding user preferences

## Advantages of CBOW

1. **Faster Training**: Averages context, making it computationally efficient
2. **Better for Frequent Words**: Works well with common words that have rich context
3. **Smoother Embeddings**: Averaging produces more stable representations

## Limitations

1. **Order Ignorance**: Doesn't consider word order in context
2. **Rare Words**: May struggle with infrequent words
3. **Single Sense**: Each word has one embedding (can't handle polysemy)

## Running the Notebook

1. **Install Dependencies**: Run the first cell to install required packages
2. **Download Data**: The notebook automatically downloads the text8 corpus
3. **Run Cells Sequentially**: Execute cells in order to build and train the model
4. **Visualize Results**: Explore embeddings through t-SNE plots and similarity searches

## Expected Results

After training, you should observe:
- **Decreasing Loss**: Training loss should decrease over epochs
- **Semantic Clustering**: Similar words cluster together in t-SNE visualization
- **Meaningful Similarities**: Similar words should have high cosine similarity (e.g., "king" similar to "queen", "man" similar to "woman")

## Further Reading

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781) - Original Word2Vec paper by Mikolov et al.
- [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) - Detailed explanation of CBOW and Skip-gram
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html) - Alternative implementation using Gensim

## Conclusion

This implementation demonstrates how neural networks can learn meaningful word representations from raw text. The CBOW model, despite its simplicity, captures rich semantic and syntactic relationships that are fundamental to modern NLP applications. The learned embeddings can be used as features for downstream tasks or analyzed directly to understand language patterns.
