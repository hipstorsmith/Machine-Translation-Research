# Machine-Translation-Research
This is a research project for IPI RAS about creating a machine learning translation software

## Corpus
Tab-delimited Bilingual Sentence Pairs (Russian - English): http://www.manythings.org/anki/

## Hardware
Google Colab instance with 25GB RAM & Tesla P100 16GB

## Software requirements
* python >= 3.6
* numpy >= 1.19.5
* matplotlib >= 3.2.2
* tensorflow == 1.15.2
* nltk >= 3.2.5

## Data (models, datasets and tokenizers)
Download: https://drive.google.com/drive/folders/1-sEg-wflHO7f32xwQqAcbcFOf3CZPrSu?usp=sharing

### Datasets
This folder contains datasets in pickle format
- english-russian.pkl: full cleaned-up dataset
- 10k/30k/100k: folders containing datasets cropped to 10000/30000/100000 pairs of sentences. english-russian-<>k-both.pkl - full dataset, english-russian-<>k-train.pkl - training data, english-russian-<>k-test.pkl - test data
Highly recommended to use 100k dataset in terms of resources-quality balance.

### Models
This folder contains trained models. Each model folder contains three versions for each cropped dataset (10k/30k/100k) and layout for 100k dataset
1) Basic model:
  Consists of single-layer (LSTM) encoder and single-layer (LSTM) decoder. Each model is trained for 40/22/13 iterations respectively
2) Basic bidirectional model:
  Consists of single-layer (Bidirectional LSTM) encoder and single-layer (LSTM) decoder. Each model is trained for 30/30/12 iterations respectively
3) Deep w. dropout:
  Consists of two-layer (Bidirectional LSTM) encoder and four-layer (LSTM) decoder. Each model is trained for 100 iterations
4) Deep w. dropout, recurrent dropout and attention mechanism:
  Consists of two-layer (Bidirectional LSTM) encoder and four-layer (LSTM) decoder. Attention mechanism takes inputs from first encoder and last decoder layer. Each model is trained for 68/52/29 iterations respectively
Highly recommended to use Model 4 with attention with 100k dataset

### Tokenizers
Contains english and russian tokenizers for 10k/30k/100k dataset

### Corpus
Original non-cleaned corpus (see above)

## Notebooks
Notebooks folder contains folders with final model (Model 4, see above) and older models. In each folder there are notebooks with training and evaluation of model

## Data preparation
1) Load corpus and split it to pairs
2) Clean punctuation and convert sentences to lowercase
3) Dump cleaned data to pkl file
4) Dataset is already sorted by number of words in english sentence so just crop dataset to required number of pairs
5) Split cropped dataset to train and test (train/full = 90%)
6) Dump cropped dataset, train and test to pkl

## Training
1) Load cropped dataset and respective train and test
2) Prepare english and russian tokenizers
3) Encode data and target sentences from train and test to sequences and pad them
4) Encode target sequences from train and test to one-hot encodings
5) Define model, optimizer (Adam) and metric (categorical crossentropy)
6) Start training. You can set saving checkpoints to saving best only

## Evaluation
1) Load datasets from pkl
2) Load tokenizers from pkl
3) Encode data sentences from train and test to sequences and pad them
4) Load model from last saved checkpoint
5) Predict target sentences using model. If data is too big to fit into memory use batch generator
6) Compare predicted sentences with target sentences and calculate BLEU and GLEU score
