# Roadmap — Byte-Level BPE Integration

## 1. Tokenizer Implementation
- [X] Implement GPT-2 byte-to-Unicode mapping.
- [X] Implement Byte-Level BPE pre-tokenization.
- [X] Implement BPE pair frequency counting and merging.
- [X] Implement vocabulary and merge rules.
- [ ] Implement text encoding (text → token IDs).
- [ ] Implement text decoding (token IDs → text).
- [ ] Support special tokens.
- [ ] Save and load tokenizer vocabulary and merge rules.

## 2. Transformer Adaptation
- [ ] Replace OHE input with integer token IDs.
- [ ] Implement token embedding lookup using an embedding matrix.
- [ ] Adapt positional embeddings to work with token ID sequences.
- [ ] Remove OHE generation from the data pipeline.
- [ ] Adapt embedding backpropagation to update only the corresponding token embeddings.
- [ ] Ensure the model output dimension matches the tokenizer vocabulary size.

## 3. Training Pipeline Adaptation
- [ ] Adapt the Parquet loader to retrieve raw text.
- [ ] Tokenize train, validation and test datasets.
- [ ] Replace OHE batches with integer token ID batches.
- [ ] Adapt target representation to token IDs.
- [ ] Update loss calculation to use integer targets (sparse cross-entropy).
- [ ] Persist tokenized datasets to avoid repeated tokenization.

## 4. Testing
- [ ] Test byte-to-Unicode mapping.
- [ ] Test BPE encoding and decoding.
- [ ] Test whitespace and leading-space preservation (`Ġ`).
- [ ] Test Unicode characters, accents and emojis.
- [ ] Test embedding lookup and backpropagation.
- [ ] Test training with token IDs instead of OHE matrices.
- [ ] Compare tokenizer outputs against GPT-2 reference implementations.

## 5. MLflow and MinIO Integration
- [ ] Log tokenizer configuration and vocabulary size.
- [ ] Store tokenizer vocabulary and merge rules as artifacts.
- [ ] Log model checkpoints and training metrics.
- [ ] Track tokenizer and dataset versions.