import torch
from torch.utils.data import RandomSampler
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from nlpkge import *
device=init_nlpkge(device_id="7",seed=1)

loader =FB15KLoader(dataset_path="../dataset",download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut= loader.load_all_lut()

processor = FB15KProcessor(node_lut, relation_lut,reprocess=True,mode="normal")
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut,relation_lut=processor.process_lut()

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = Rescal(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

loss = MarginLoss(margin=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

metric = Link_Prediction(node_lut=node_lut,
                         relation_lut=relation_lut,
                         link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=1000000,
                         reverse=False)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
    factor=0.5, min_lr=1e-9, verbose=True
)

negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                       entity_dict_len=len(node_lut),
                                       relation_dict_len=len(relation_lut),
                                       node_lut=node_lut)

trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    test_sampler=test_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    output_path="../dataset",
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
    metric=metric,
    trainer_batch_size=2000000,
    total_epoch=2,
    lr_scheduler=lr_scheduler,
    apex=True,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    use_tensorboard_epoch=100,
    use_matplotlib_epoch=100,
    use_savemodel_epoch=100,
    use_metric_epoch=1
)
trainer.train()


