# order-learning

### Dataset Assumptions
For a problem that requires an order to be learnt, availability of the following two datasets is assumed: 

* $D_{main}$ : A dataset of $x_i$ which ultimately we would like to predict the order between any two members. 
* $D_{comp}$ : A dataset consisting of ordered tuples $(x_i, x_j, o)$, where $o \in {\geq, \leq, \sim }$

### Ranking Color Gradient

A simple toy example. 
$D_{main}$ is made up of 40 different gradients of green, rank 1 is the darkest and rank 40 is the lightest. 

| Rank = 1  | Rank = 10 | Rank = 20 | Rank = 30 | Rank = 40 | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ![](assets/0.png) | ![](assets/10.png) | ![](assets/20.png) | ![](assets/30.png) | ![](assets/39.png) |


For $D_{comp}$ we create two different sampling methods. 
EXTREMES: greater than has rank > 30 and lesser than has rank < 10
CENTER: greater than has 30 > rank > 20 and lesser than has 10 < rank < 20

code for these datasets is [here](datasets.py)