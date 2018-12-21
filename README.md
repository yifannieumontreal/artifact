# MACM

This in the implementation of the representation- and interaction-based Multi-level Abstraction Convolutional Model (rep_MACM and inter_MACM)


## prerequisites

* Python 3.6.3
* Tensorflow 1.3.1
* Pytorch 0.3
* Numpy


## Usage

### Configure: There are 2 types of models, one is representation-based model, the other is interaction-based model
first, configure the hyperparameter through the config file, a sample is provided

[sample_rep.config](https://github.com/yifannieumontreal/artifact/blob/master/sample_inter.config)
[sample_inter.config](https://github.com/yifannieumontreal/artifact/blob/master/sample_rep.config)

A folder for saving models should be created on local disk and should contain the config file and 4 sub-folders:

```
model
..\config
..\logs
..\result
..\saves
..\tmp

```

### Train

To train the representation-based model, pass the config file path, mode, and resume flag into command line
```
python rep_MACM_train.py --path: path to the config file \
--mode train  \
--resume False  (whether to resume training from the saved model or train a brand new model)
```

To train the interaction-based model, pass the config file path, mode, and resume flag into command line
```
python inter_MACM_train.py --path: path to the config file \
--mode train  \
--resume False  (whether to resume training from the saved model or train a brand new model)
```

### Test
To test the representation-based model, pass the config file path, mode into command line
```
python rep_MACM_train.py --path: path to the config file \
--mode test
```

To test the interaction-based model, pass the config file path, mode into command line
```
python inter_MACM_train.py --path: path to the config file \
--mode test
```

## Data Preprocessing
All queries and documents should be encoded into sequences of integer term ids, term id should begin with 1, where 1 indicates OOV term.
Training data should be stored in python dict with the following structure:
```
data = {
   qid:{'query': [257, 86, 114],
        'docs': [[123, 456, 6784...], [235, 345, 768,...],...]
        'scores': [25.16, 16.83, ...]
   }
}
```
qid should be a str type, e.g. '31'

Validation or testing data should be stored in python dict with the following structure:
```
test_data = {
  qid:{'query': [257, 86, 114],
       'docs': [[123, 456, 6784...], [235, 345, 768,...],...]
       'docno': ['clueweb09-en0000-00-00000', 'clueweb09-en0000-00-00001'...]
  }
}
``` 
qid should be a str type, e.g. '51'

A set of pre-trained embeddings should also be specified to input into the model. In this paper we employed [GloVe6B.300d](https://nlp.stanford.edu/projects/glove/), you can also train models like word2vec on your own corpus. The embedding data should be a pickled file containging an embedding matrix of shape [vocab_size, emb_dim] of data type float32, and stored under the base_data_path (configured in the config file).

## Configurations
### For representation-based model
*model_name_str: model folder's name
*batch_size: batch size
*vocab_size: vocabulary size
*emb_size: embedding size
*hidden_size: hidden size
*dropout: dropout rate
*q_filt1: first conv filter size for query
*q_filt2: second layer conv filter size for query
*q_filt3: third layer conv filter size for query
*q_stride1: stride for the first query conv layer
*q_stride2: stride for the 2nd query conv layer
*q_stride3: stride for the 3rd query conv layer
*d_filt1: filter size for the 1st layer of document conv
d_filt2:  filter size for the 2nd layer of document conv
d_filt3:  filter size for the 3rd layer of document conv
d_stride1:  stride for the 1st layer of document conv
d_stride2:  stride for the 2nd layer of document conv
d_stride3:  stride for the 3rd layer of document conv
Mtopk: top K values to take for the M value
preemb: bool, use pretrained embeddings or not
preemb_path: path to the pre-trained embeddings
sim_type: similarity metric
hinge_margin: hinge margin for the loss function
train_datablock_size: training data block size to store a block of training examples in RAM during training
q_sample_size: training query sampling size if using sampling to sample a subset of training queries
docpair_sample_size: document pair sampling size if using sampling to sample a subset of training samples
n_epoch: max num of epochs
alpha: L2 normalization weight
learning_rate: learning rate
q_len: max query length (num of query terms)
d_len: max document length (num of query terms)
data_base_path: full path to the parent folder of the model folder (i.e. if model folder is located in /scratch/models/macm, this base path should be /scratch/models/)
model_base_path: full path to the parent folder of the training data folder (i.e. if the train data folder is located in /scratch/data/train/, this base path should be /scratch/data/)

### For interaction-base model
* model_name_str: model folder's name
* batch_size: batchsize
* vocab_size: vocabulary size
* emb_size: embedding size
* n_gramsets: num of parallel convolutions with different filter size for each conv layer, should be set to 1 for the MACM model
* n_filters1: num of filters for the first conv layer, should be in brackets, e.g. [32]
* n_filters2: num of filters for the second conv layer, should be in brackets, e.g. [16]
* kernel_sizes1: filter shape of conv layer 1
* kernel_sizes2: filter shape of conv layer 2
* conv_strides1: stride for the 1st conv layer
* conv_strides2: stride for the 2nd conv layer
* pool_sizes0: pooling size for the interaction matrix
* pool_sizes1: pooling size for the 1st conv layer
* pool_sizes2: pooling size for the 2nd conv layer
* pool_strides: stride for all pooling operactions
* n_hiddden_layers: num of hidden layers in the flattened MLP after each conv layer
* hidden_sizes: hidden size for mlp hidden layers
* hinge_margin: hinge margin for pairwise training loss
* train_datablock_size: training data block size to store a block of training examples in RAM during training
* q_sample_size: training query sampling size if using sampling to sample a subset of training queries
* docpair_sample_size: document pair sampling size if using sampling to sample a subset of training samples
* n_epoch: max num of epochs
* alpha: L2 normalization weight
* q_len: max query length (num of query terms)
* d_len: max document length (num of document terms)
* model_base_path: full path to the parent folder of the model folder (i.e. if model folder is located in /scratch/models/macm, this base path should be /scratch/models/)
* data_base_path: full path to the parent folder of the training data folder (i.e. if the train data folder is located in /scratch/data/train/, this base path should be /scratch/data/)


