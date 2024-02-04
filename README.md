# Gesture2Vec: Clustering Gestures using Representation Learning Methods for Co-speech Gesture Generation

## JH instructions (February 2024)

Create the environment:
```bash
conda env create -f gesture2vec.yml
conda activate gesture2vec
```

### Preprocess Trinity data

Download fastText vectors:
```bash
mkdir resource
cd resource
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
```

Make LMDB data for training set:
```bash
cd scripts
python trinity_data_to_lmdb.py /n/holylabs/LABS/kempner_fellows/Users/jennhu/GENEA_Challenge_2020_data_release/Training_data
```
This will create `lmdb_train` and `lmdb_test`, which should be treated as training and validation sets, respectively.

### Train DAE (frame-level model)
```bash
python train_DAE.py --config=../config/DAE_GENEA_jh.yml
```
Or submit the SLURM script:
```bash
sbatch train_DAE.batch
```

## The Best Paper Award Winner in Cognitive Robotics at IROS2022

This is an official PyTorch implementation of _Gesture2Vec: Clustering Gestures using Representation Learning Methods for Co-speech Gesture Generation_ (IROS 2022). In this paper, we present an automatic gesture generation model that uses a vector-quantized variational autoencoder structure as well as training techniques to learn a rigorous representation of gesture sequences. We then translate input text into a discrete sequence of associated gesture chunks in the learned gesture space. Subjective and objective evaluations confirm the success of our approach in terms of appropriateness, human-likeness, and diversity. We also introduce new objective metrics using the quantized gesture representation.

### [Paper](https://sfumars.com/wp-content/papers/2022_iros_gesture2vec.pdf) | [Demo Video](https://www.youtube.com/watch?v=ac8jWk4fdCU) | [Presentation](https://youtu.be/qFObMpOboCg)

![OVERVIEW](Figures/model.jpg)

## Demo Video

[![Demo Video](https://img.youtube.com/vi/ac8jWk4fdCU/0.jpg)](https://www.youtube.com/watch?v=ac8jWk4fdCU)

## Presentation

[![IROS2022 Presentation](https://img.youtube.com/vi/qFObMpOboCg/0.jpg)](https://www.youtube.com/watch?v=qFObMpOboCg)

## Instructions

TODO

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code uses datasets inluding Trinity and Talk With Hand (TWH) that each have their own respective licenses that must also be followed.

Please feel free to contact us (pjomeyaz@sfu.ca) with any question or concerns.
