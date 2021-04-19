# The FacT

This is an implementation for the paper titled "The FacT: Taming Latent Factor Models for Explainability with Factorization Trees" which is published at SIGIR 2019. We public our source code in this repository.

### Algorithm
PairRank aims at learning a pairwise learning to rank model online. In each round, candidate documents are partitioned and ranked according to the model's confidence on the estimated pairwise rank order, and exploration is only performed on the uncertain pairs of documents, i.e., divide-and-conquer.  
Regret directly defined on the number of mis-ordered pairs is proven, which connects the online solution's theoretical convergence with its expected ranking performance.

### Usage
To run the code to generate experimental results like those found in our papers, you will need to run a command in the following format, using Python 2:
```
$ python run_script.py [--algo ALGO] [--data_sets DATASET] [--lr LEARNING_RATE] [--lr_decay LEARNING_RATE_DECAY]
                       [--lambda LAMBDA] [--alpha ALPHA] [--click_models CLICK_MODEL] [--n_impr N_IMPR] [--update UPDATE]
                       [--rank RANK] [--ind] [--refine] [--seed SEED] 
```
Example:
```
$ python run_script.py --algo pairrank --data_sets yahoo --click_models per --n_impr 5000 --lambda 0.1 --alpha 0.1 --update gd --rank random --seed 0 --ind --refind
```
The detailed information about the input parameters can be found in /utils/argparsers/simulationargparser.py 

In our paper, we have two widely used benchmark datasets: [Yahoo! Learning to Rank dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c) and [MSLR-WEB10K dataset](https://www.microsoft.com/en-us/research/project/mslr/).



