# Loss Zoo

## Available Loss Functions

### SIGReg (Sketched Isotropic Gaussian Regularization)

**Source**: [LeJEPA](https://arxiv.org/abs/2511.08544)

**Use Case**: Embedding regularization in self-supervised learning. Tests whether embeddings follow a standard normal distribution N(0,I) using characteristic functions.

**Benefits**: Replaces traditional heuristics like stop-gradient and teacher-student mechanisms with theoretical guarantees. Core implementation is only ~50 lines with a single hyperparameter. Loss value highly correlates with downstream task performance (94%+ Spearman), enabling model selection without labeled validation data. Works out-of-the-box with 60+ architectures (ResNets, ViTs, ConvNeXt) without hyperparameter tuning.
