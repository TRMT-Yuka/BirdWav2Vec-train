# BirdWav2Vec-train

This training code is based on 
transformers/examples/pytorch/speech-pretraining
in https://github.com/huggingface/transformers

- `check.py` checks a target model and dataset
- `extract_result.py`  creates `result.pkl`, storeing embedding vectors from all audio samples in a target dataset
- `plot_result.py`: plots embeding space from `result.pkl`
- `model_push_to_hub.py`: pushes model to huggingface
 
## Training (pretraining)

```
sh run_birddb.sh
```
This script performs speech-pretraining for bird songs 
using `run_wav2vec2_pretraining_no_trainer.py`.


## 【追記】

run_birddb.shにおいて,dataset_nameとして指定されるデータセットが非公開の場合に対応

.envファイルを作成し, 

```
huggingface_TOKEN=hf_TcViexample.....(hugging faceのアクセストークンを記入)
```
