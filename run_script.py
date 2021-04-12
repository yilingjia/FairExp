from algorithms.Pairwise.PairRank import PairRank
from algorithms.Pairwise.PairDiff import PairDiff
from algorithms.Pairwise.PairDiff_2 import PairDiff_2
from algorithms.Pairwise.NeuralPairRank import NeuralPairRank
from algorithms.Pairwise.NeuralPairRank_0 import NeuralPairRank_0
from algorithms.Pairwise.LambdaPairRank import LambdaPairRank
from algorithms.Pairwise.RandomRankNet import RandomRankNet
from algorithms.Pairwise.RandomLambdaRank import RandomLambdaRank
from algorithms.Pairwise.NeuralGT import NeuralGT
from algorithms.Pairwise.LambdaGT import LambdaGT
from utils.argparsers.simulationargparser import SimulationArgumentParser
from utils.datasimulation import DataSimulation

# from utils.gpu_profile import gpu_profile
# import sys

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["GPU_DEBUG"] = "1"
# CUDA_VISIBLE_DEVICES=1
# import sys
# import datetime


def func_pairrank(args, dir_name):
    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "refine": args.refine,
        "rank": args.rank,
        "ind": args.ind,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)
    if args.update == "gd" or args.update == "gd_diag" or args.update == "gd_recent":
        ranker_name = "None-None-{}-{}-{}-{}-{}-{}-{}".format(
            args.update, args.lmbd, args.alpha, args.refine, args.rank, args.ind, args.seed,
        )
    else:
        ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
            args.lr, args.lr_decay, args.update, args.lmbd, args.alpha, args.refine, args.rank, args.ind, args.seed,
        )

    run_name = dir_name + ranker_name
    ranker = [(run_name, PairRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_pairdiff_2(args, dir_name):
    # use the pointwise feature to update matrix A
    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "rank": args.rank,
        "ind": args.ind,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)
    if args.update == "gd" or args.update == "gd_diag" or args.update == "gd_recent":
        ranker_name = "None-None-{}-{}-{}-{}-{}-{}".format(
            args.update, args.lmbd, args.alpha, args.rank, args.ind, args.seed,
        )
    else:
        ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(
            args.lr, args.lr_decay, args.update, args.lmbd, args.alpha, args.refine, args.rank, args.seed,
        )

    run_name = dir_name + ranker_name
    ranker = [(run_name, PairDiff_2, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_pairdiff(args, dir_name):
    # use the pairwise feature to update the matrix A
    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "rank": args.rank,
        "ind": args.ind,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)
    if args.update == "gd" or args.update == "gd_diag" or args.update == "gd_recent":
        ranker_name = "None-None-{}-{}-{}-{}-{}-{}".format(
            args.update, args.lmbd, args.alpha, args.rank, args.ind, args.seed,
        )
    else:
        ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(
            args.lr, args.lr_decay, args.update, args.lmbd, args.alpha, args.refine, args.rank, args.seed,
        )

    run_name = dir_name + ranker_name
    ranker = [(run_name, PairDiff, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_neuralgt(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}".format(
        args.lr, args.lr_decay, args.update, args.lmbd, args.batch_size, args.epoch, args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, NeuralGT, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_lambdagt(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}".format(
        args.lr, args.lr_decay, args.update, args.lmbd, args.batch_size, args.epoch, args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, LambdaGT, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_neuralpairrank(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "refine": args.refine,
        "rank": args.rank,
        "ind": args.ind,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.lr,
        args.lr_decay,
        args.update,
        args.lmbd,
        args.alpha,
        args.refine,
        args.rank,
        args.ind,
        args.batch_size,
        args.epoch,
        args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, NeuralPairRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_neuralpairrank_0(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "refine": args.refine,
        "rank": args.rank,
        "ind": args.ind,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.lr,
        args.lr_decay,
        args.update,
        args.lmbd,
        args.alpha,
        args.refine,
        args.rank,
        args.ind,
        args.batch_size,
        args.epoch,
        args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, NeuralPairRank_0, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_lambdapairrank(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "alpha": args.alpha,
        "refine": args.refine,
        "rank": args.rank,
        "ind": args.ind,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.lr,
        args.lr_decay,
        args.update,
        args.lmbd,
        args.alpha,
        args.refine,
        args.rank,
        args.ind,
        args.batch_size,
        args.epoch,
        args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, LambdaPairRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_randomranknet(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "ind": args.ind,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": args.nnlayers,
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.lr, args.lr_decay, args.update, args.lmbd, args.ind, args.batch_size, args.epoch, args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, RandomRankNet, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def func_randomlambdarank(args, dir_name):

    ranker_params = {
        "learning_rate": args.lr,
        "learning_rate_decay": args.lr_decay,
        "update": args.update,
        "_lambda": args.lmbd,
        "ind": args.ind,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "mlp_dims": [64, 32],
    }
    sim_args, other_args = parser.parse_all_args(ranker_params)

    ranker_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.lr, args.lr_decay, args.update, args.lmbd, args.ind, args.batch_size, args.epoch, args.seed,
    )

    run_name = dir_name + ranker_name
    ranker = [(run_name, RandomLambdaRank, other_args)]
    sim = DataSimulation(sim_args)
    sim.run(ranker)


def set_sim_and_run(args):
    cm = args.click_models[0]
    n_impr = args.n_impressions
    n_results = args.n_results
    algo = args.algo.upper()
    nn_layer = args.nnlayers
    layer_folder = "_".join([str(layer) for layer in nn_layer])
    dir_name = "algo/{}/{}/{}/{}/{}/".format(algo, cm, n_impr, n_results, layer_folder)

    switcher = {
        "PAIRRANK": lambda: func_pairrank(args, dir_name),
        "PAIRDIFF": lambda: func_pairdiff(args, dir_name),
        "PAIRDIFF_2": lambda: func_pairdiff_2(args, dir_name),
        "NEURALPAIRRANK": lambda: func_neuralpairrank(args, dir_name),
        "NEURALPAIRRANK_0": lambda: func_neuralpairrank_0(args, dir_name),
        "LAMBDAPAIRRANK": lambda: func_lambdapairrank(args, dir_name),
        "RANDOMRANKNET": lambda: func_randomranknet(args, dir_name),
        "RANDOMLAMBDARANK": lambda: func_randomlambdarank(args, dir_name),
        "NEURALGT": lambda: func_neuralgt(args, dir_name),
        "LAMBDAGT": lambda: func_lambdagt(args, dir_name),
    }

    return switcher.get(algo, lambda: "ERROR: algorithm type not valid")()


# get input parameters
if __name__ == "__main__":
    # sys.settrace(gpu_profile)
    DESCRIPTION = "Run script for testing framework."
    parser = SimulationArgumentParser(description=DESCRIPTION)
    input_args = parser.parse_sim_args()
    set_sim_and_run(input_args)
