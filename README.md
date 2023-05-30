# BirdWav2Vec-train

This training code is based on 
transformers/examples/pytorch/speech-pretraining
in https://github.com/huggingface/transformers

- `check.py` checks a target model and dataset
- `extract_result.py`  creates `result.pkl`, storeing embedding vectors from all audio samples in a target dataset
- `plot_result.py`: plots embeding space from `result.pkl`
- `model_push_to_hub.py`: pushes model to huggingface
 
