from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining)
 


processor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base-v2")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./wav2vec2-birddb")
#config = Wav2Vec2Config.from_pretrained("wav2vec2-birddb")
#model = Wav2Vec2ForPreTraining(config)
model = Wav2Vec2ForPreTraining.from_pretrained("./wav2vec2-birddb")
#print(config)
print(model)

import datasets
from datasets import load_dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    if len(audio["array"])>=44100*0.1:
        return True
    return False
name="kojima-r/birddb_small"
dataset = load_dataset(name, split='train')
dataset = dataset.filter(prepare_dataset)

dataset = dataset.cast_column(
    "audio", datasets.features.Audio(sampling_rate=processor.sampling_rate)
    )

print(len(dataset))
print(dataset[0])


