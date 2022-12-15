## Description   
我们开发了一个知识图嵌入工具包，我们称之为 NlpKGE ，旨在表示多源和异构知识。 
目前，我们的demo支持 17 个模型、11 个数据集，
包括两个多源异构 KG、五个评估指标、四个知识适配器、
四个损失函数、三个采样器和三个内置数据容器。

我们提出的 python 工具包具有以下优点：

多源和异构知识表示。我们研究了来自不同来源的知识的统一表示。 
此外，我们的工具包不仅包含基于事实的三重嵌入模型，
还支持附加信息的融合表示，包括文本描述、节点类型和时间信息。

综合模型和基准数据集。我们的工具包在平移距离模型、语义匹配模型、
基于图神经网络的模型和基于transformer的模型四大类中实现了很多经典的KGE模型。 
除了开箱即用的模型，我们还使用了两个大型基准数据集，
用于进一步评估 KGE 方法，称为 EventKG240K 和 CogNet360K。

可扩展和模块化的框架。我们的工具包为 KGE 任务提供了一个编程框架。 
基于可扩展架构，我们的工具包可以满足模块扩展和二次开发的需求，
预训练的知识嵌入可以直接应用于下游任务。

开源和可视化演示。 除了工具包，我们还发布了一个在线系统，
以可视化方式发现知识。 源代码、数据集和预训练嵌入是公开可用的。
## Install 

### Install from git

```bash
# clone NlpKGE   
git clone https://github.com/heng840/NlpKGE

# install NlpKGE   
cd nlpkge
pip install -e .   
pip install -r requirements.txt
```
### Install from pip

```bash
pip install nlpkge
```

## Quick Start

### Pre-trained Embedder for Knowledge Discovery

```python
from nlpkge import *

# loader lut
device = init_nlpkge(device_id="0", seed=1)
loader = EVENTKG240KLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut, time_lut = loader.load_all_lut()
processor = EVENTKG240KProcessor(node_lut, relation_lut, time_lut,
                               reprocess=True,
                               type=False, time=False, description=False, path=False,
                               time_unit="year",
                               pretrain_model_name="roberta-base", token_len=10,
                               path_len=10)
node_lut, relation_lut, time_lut = processor.process_lut()

# loader model
model = BoxE(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

# load predictor
predictor = Predictor(model_name="BoxE",
                      data_name="EVENTKG2M",
                      model=model,
                      device=device,
                      node_lut=node_lut,
                      relation_lut=relation_lut,
                      pretrained_model_path="data/BoxE_Model.pkl",
                      processed_data_path="data",
                      reprocess=False,
                      fuzzy_query_top_k=10,
                      predict_top_k=10)

# fuzzy query node
result_node = predictor.fuzzy_query_node_keyword('champion')
print(result_node)

# fuzzy query relation
result_relation = predictor.fuzzy_query_relation_keyword("instance")
print(result_relation)

# query similary nodes
similar_node_list = predictor.predict_similar_node(node_id=0)
print(similar_node_list)

# given head and relation, query tail
tail_list = predictor.predcit_tail(head_id=0, relation_id=0)
print(tail_list)

# given tail and relation, query head
head_list = predictor.predict_head(tail_id=0, relation_id=0)
print(head_list)

# given head and tail, query relation
relation_list = predictor.predict_relation(head_id=0, tail_id=0)
print(relation_list)

# dimensionality reduction and visualization of nodes
visual_list = predictor.show_img(node_id=100, visual_num=1000)
```

### Programming Framework for Training Models

```python
import torch
from torch.utils.data import RandomSampler
from nlpkge import *

device = init_nlpkge(device_id="0", seed=1)

loader = EVENTKG240KLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut, time_lut = loader.load_all_lut()

processor = EVENTKG240KProcessor(node_lut, relation_lut, time_lut,
                               reprocess=True,
                               type=True, time=False, description=False, path=False,
                               time_unit="year",
                               pretrain_model_name="roberta-base", token_len=10,
                               path_len=10)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut, relation_lut, time_lut = processor.process_lut()

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = TransE(entity_dict_len=len(node_lut),
               relation_dict_len=len(relation_lut),
               embedding_dim=50)

loss = MarginLoss(margin=1.0, C=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

metric = Link_Prediction(link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=5000000,
                         reverse=False)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
    factor=0.5, min_lr=1e-9, verbose=True
)

negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                       entity_dict_len=len(node_lut),
                                       relation_dict_len=len(relation_lut))

trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    output_path="../dataset",
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
    metric=metric,
    lr_scheduler=lr_scheduler,
    log=True,
    trainer_batch_size=100000,
    epoch=3000,
    visualization=1,
    apex=True,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    metric_step=200,
    save_step=200,
    metric_final_model=True,
    save_final_model=True,
    load_checkpoint=None
)
trainer.train()

evaluator = Evaluator(
    test_dataset=test_dataset,
    test_sampler=test_sampler,
    model=model,
    device=device,
    metric=metric,
    output_path="../dataset",
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
    log=True,
    evaluator_batch_size=50000,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    trained_model_path=None
)
evaluator.evaluate()
```

## Model

<table class="greyGridTable" >
    <thead>
        <tr >
            <th >Category</th>
            <th >Model</th>
            <th>Conference</th>
            <th>Paper</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7" >Translation Distance Models</td>
            <td>
                <a href="https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf">TransE</a> 
            </td>
            <td>NIPS 2013</td>
            <td>Translating embeddings for modeling multi-relational data</td>
        </tr>
        <tr>
            <td>
                <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf">TransH</a> 
            </td>
            <td>AAAI 2014</td>
            <td>Knowledge Graph Embedding by Translating on Hyperplanes</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523">TransR</a> 
            </td>
            <td>AAAI 2015</td>
            <td>Learning Entity and Relation Embeddings for Knowledge Graph Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.aclweb.org/anthology/P15-1067.pdf">TransD</a> 
            </td>
            <td>ACL 2015</td>
            <td>Knowledge Graph Embedding via Dynamic Mapping Matrix</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1509.05490.pdf">TransA</a> 
            </td>
            <td>AAAI 2015</td>
            <td>TransA: An Adaptive Approach for Knowledge Graph Embedding</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/2007.06267.pdf">BoxE</a> 
            </td>
            <td>NIPS 2020</td>
            <td>BoxE: A Box Embedding Model for Knowledge Base Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/2011.03798.pdf">PairRE</a> 
            </td>
            <td>ACL 2021</td>
            <td>PairRE: Knowledge Graph Embeddings via Paired Relation Vectorss</td>
        </tr>
        <tr>
            <td rowspan="5">Semantic Matching Models</td>
            <td>
                <a href="https://icml.cc/2011/papers/438_icmlpaper.pdf">RESCAL</a> 
            </td>
            <td>ICML 2011</td>
            <td>A Three-Way Model for Collective Learning on Multi-Relational Data</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1412.6575.pdf">DistMult</a> 
            </td>
            <td> ICLR 2015</td>
            <td>Embedding Entities and Relations for Learning and Inference in Knowledge Bases</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1802.04868.pdf">SimplE</a> 
            </td>
            <td>NIPS 2018</td>
            <td>SimplE Embedding for Link Prediction in Knowledge Graphs</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1901.09590.pdf">TuckER</a> 
            </td>
            <td>ACL 2019</td>
            <td>TuckER: Tensor Factorization for Knowledge Graph Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1902.10197.pdf">RotatE</a> 
            </td>
            <td>ICLR 2019</td>
            <td>RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space</td>
        </tr>
        <tr>
            <td rowspan="2">Graph Neural Network-based Models</td>
            <td>
                <a href="https://arxiv.org/pdf/1703.06103.pdf">R-GCN</a> 
            </td>
            <td>ESWC 2018</td>
            <td>Modeling Relational Data with Graph Convolutional Networks</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1911.03082.pdf">CompGCN</a> 
            </td>
            <td>ICLR 2020</td>
            <td>Composition-based Multi-Relational Graph Convolutional Networks</td>
        </tr>
        <tr>
            <td rowspan="2">Transformer-based Models</td>
            <td>
                <a href="https://arxiv.org/pdf/2008.12813.pdf">HittER</a> 
            </td>
            <td>EMNLP 2021</td>
            <td>HittER: Hierarchical Transformers for Knowledge Graph Embeddings</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.researchgate.net/profile/Jian-Tang-46/publication/337273572_KEPLER_A_Unified_Model_for_Knowledge_Embedding_and_Pre-trained_Language_Representation/links/6072896c299bf1c911c2051a/KEPLER-A-Unified-Model-for-Knowledge-Embedding-and-Pre-trained-Language-Representation.pdf">KEPLER</a> 
            </td>
            <td>TACL 2021</td>
            <td>KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation</td>
        </tr>
    </tbody>
</table>

## Dataset

### [EventKG240K](https://eventkg.l3s.uni-hannover.de/)
EventKG 是一个以事件为中心的时态知识图谱，包含超过 69 万个当代和历史事件以及超过 230 万个时态关系。 据我们所知，EventKG240K 是第一个以事件为中心的 KGE 数据集。 我们使用 EventKG V3.0 数据构建数据集。
首先，我们根据程度过滤实体和事件。 然后，当两个节点的度都大于10时，我们选择三元组事实。最后，我们为节点添加文本描述和节点类型，并通过时间信息将三元组转换为四元组。 整个数据集包含 238,911 个节点、822 个关系和 2,333,986 个三元组。
### [CogNet360K](http://cognet.top/)
CogNet 是一个多源异构 KG，致力于整合语言、世界和常识知识。 为了构建一个子集作为数据集，我们计算每个节点的出现次数。 然后，我们根据连接节点的最小出现次数对框架实例进行排序。 有了排序列表后，我们根据预设的框架类别过滤三重事实。 最后，我们找到参与这些三重事实的节点并完成它们的信息。 最终数据集包含 360,637 个节点和 1,470,488 个三元组。
## Other KGE open-source project

 - [Graphvite](https://graphvite.io/)
 - [OpenKE](https://github.com/thunlp/OpenKE)
 - [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN)
 - [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
 - [LIBKGE](https://github.com/uma-pi1/kge)
 - [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
 - [PyG](https://github.com/pyg-team/pytorch_geometric)
 - [CogDL](https://github.com/THUDM/cogdl)
 
