## ESTeR: Emotion-Sensitive TextRank

The repo contains implementations, datasets, and other resources used for a our paper: 

*ESTeR: Combining Word Co-occurrences and Word Associations for Unsupervised Emotion Detection. Sujatha Das Gollapalli, Polina Rozenshtein, See-Kiong Ng. Findings of EMNLP 2020.*

The datasets and resources, which are not freely available, are excuded.

= Python 3.6 =

* list of dependencies `requirements.txt`
* installation: `pip3 install -r requirements.txt`
* for optimal running time ensure that BLAS/LAPACK is installed and used in NumPy and SciPy

### main scripts
* `run.py` runs any of the scoring algorithms on the specified datasets and resources, reports quality of the classification.
```
usage: run.py [-h] [-d DATASET] [-l LEXICON] [-c COOC] [-e EMBEDDING]
              [-m METHOD] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        datasets: semval2018, crowdflower, ssec, tec, dens.
                        (default: semval2018)
  -l LEXICON, --lexicon LEXICON
                        lexicons: NRC-EmoLex, DepecheMood. (default: NRC-
                        EmoLex)
  -c COOC, --cooc COOC  cooc matrices: wiki, twitter, comb. (default: comb)
  -e EMBEDDING, --embedding EMBEDDING
                        embeddings: emo2vec100, ewe-uni300, sswe-u50.
                        (default: sswe-u50)
  -m METHOD, --method METHOD
                        methods: ester, ester_lexnorm, ester_lex2sent,
                        cos_embedding, cos_lexicon. (default: ester)
  --output OUTPUT, -o OUTPUT
                        file name to save the result score. If specified, then
                        the result is stored to OUTPUT directory from
                        config.ini file. (default: )
```
* `algorithms.py` implements scoring algorithms including baselines.
* `metrics.py` implements binary classification @k and calculations of basic classification quality metrics.

#### case_study directory
* `casestudy.py` scripts for the case study on COVID19 tweets dataset (`raw_data/COVIDtweets`)

#### datasets directory
* preprocessed labeled datasets

#### resources directory
* preprocessed lexicons, co-occurrence matrices, and embeddings

#### preprocessing directory
* scripts for preprocessing of the datasets, lexicons, and co-occurrence matrices. Use `--help` for arguments.

#### raw_data directory
* unprocessed datasets and resources
