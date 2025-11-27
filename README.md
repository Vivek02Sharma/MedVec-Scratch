# MedVec-Scratch
**MedVec-Scratch** is a custom sentence embedding model architected purely in PyTorch without relying on pre-trained backbones. It features a custom BPE tokenizer and a Siamese Transformer Encoder trained on a 233k medical triplet dataset. The model utilizes Mean Pooling and Triplet Margin Loss to achieve high-quality semantic retrieval for clinical queries.

![MedVecArchitecture](./assets/MedVec-scratch.png)

Colab-Notebook-link - https://colab.research.google.com/drive/1apmVMRNxpeVr3H51ndK1Kn5OEAvQyM5p?usp=sharing

Dataset_url - https://huggingface.co/datasets/abhinand/MedEmbed-training-triplets-v1

## Key Features:
- **Architecture**: Custom `TransformerEncoder` (4 layers, 4 heads, 256 dim) mimicking BERT-style architecture.
- **Tokenization**: Custom-trained `Byte-Pair Encoding (BPE)` tokenizer optimized for complex medical terminology (30k vocab).
- **Training Strategy**: Trained on the **MedEmbed** dataset (233k triplets) using `Triplet Margin Loss` to learn semantic similarity between medical queries and clinical answers.
- **Pooling**: Implements **Mean Pooling** with attention masking for robust sentence-level vector generation.
- **Stack**: Pure PyTorch, HuggingFace Datasets, Tokenizers, and Google Drive integration for checkpointing.

**Use Case**: Designed to power RAG (Retrieval Augmented Generation) systems and Semantic Search engines in healthcare applications.
