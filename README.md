# DHCL: Multi-Level Dual-Graph Contrastive Learning Based on Dynamic Hypergraph Optimization for Multimodal Conversational Emotion Recognition

PyTorch implementation for the paper:

**DHCL: Multi-Level Dual-Graph Contrastive Learning Based on Dynamic Hypergraph Optimization for Multimodal Conversational Emotion Recognition**

---

### Requirements

- Python 3.9
- PyTorch 1.12.1
- CUDA 11.3
- torch-geometric



### Dataset

The raw datasets can be obtained from:

- IEMOCAP: https://sail.usc.edu/iemocap/
- MELD: https://github.com/SenticNet/MELD

We use pre-extracted multimodal features:

- Text: RoBERTa
- Audio: OpenSMILE
- Visual: DenseNet

Please place the processed features under the corresponding dataset directory before training.

---

###  Checkpoints

Pretrained checkpoints will be released upon publication.

---



### Main Components

- Dual-Graph Representation Learning
- Attention-based Dynamic Hypergraph Optimization (ADHO)
- Multi-Level Contrastive Learning
  - Inter-Graph Contrastive Learning
  - Intra-Graph Contrastive Learning
  - Cross-Dialogue Contrastive Learning

---

### Experimental Results

| Dataset | Acc | Wa-F1 |
|----------|----------|----------|
| IEMOCAP | 71.90 | 71.75 |
| MELD | 67.95 | 67.04 |

---

## Citation
