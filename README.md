# Recommendation System 
#### recommender systems are composed of two stages:
- The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.
- The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

### Retrieval models are composed of two sub-models:

- A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.
- A candidate model computing the candidate representation (an equally-sized vector) using the candidate features
The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.

#### we're going to build and train such a two-tower model using the Movielens dataset.

We're going to:

- Get our data and split it into a training and test set.
- Implement a retrieval model.
- Fit and evaluate it.
- Export it for efficient serving by building an approximate nearest neighbours (ANN) index.
## Installation

Make sure you have TensorFlow 2.x installed, and install from `pip`:

```shell
pip install -r requirements.txt
```

## Quick Run
```shell
python retrival_fit.py
python scaan.py
```

## Next Step

