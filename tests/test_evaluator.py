import sys
import torch
from pathlib import Path
from torch.utils.data import RandomSampler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nlpkge import *
device=init_nlpkge(device_id="7",seed=1)

loader = EVENTKG240KLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut, time_lut = loader.load_all_lut()

processor = EVENTKG240KProcessor(node_lut, relation_lut, time_lut, reprocess=True,mode="normal")
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut, relation_lut, time_lut = processor.process_lut()

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = DistMult(entity_dict_len=len(node_lut),
                 relation_dict_len=len(relation_lut),
                 embedding_dim=50,
                 penalty_weight=0.1)

loss = NegLogLikehoodLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

metric = Link_Prediction(link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=5000000,
                         reverse=True)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
    factor=0.5, min_lr=1e-9, verbose=True
)

negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                       entity_dict_len=len(node_lut),
                                       relation_dict_len=len(relation_lut))

evaluator = Evaluator(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    test_dataset=test_dataset,
    test_sampler=test_sampler,
    checkpoint_path=None,  # checkpoint
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
    metric=metric,
    lr_scheduler=lr_scheduler,
    apex=True,
    dataloaderX=True,
    num_workers=1,
    pin_memory=True,
)
evaluator.evaluate()

