from algorithms.Pairwise.PairRank import PairRank
from utils.argparsers.simulationargparser import SimulationArgumentParser
from utils.datasimulation import DataSimulation


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


def set_sim_and_run(args):
    cm = args.click_models[0]
    n_impr = args.n_impressions
    n_results = args.n_results
    algo = args.algo.upper()
    dir_name = "algo/{}/{}/{}/{}/".format(algo, cm, n_impr, n_results)

    switcher = {
        "PAIRRANK": lambda: func_pairrank(args, dir_name),
    }

    return switcher.get(algo, lambda: "ERROR: algorithm type not valid")()


# get input parameters
if __name__ == "__main__":
    DESCRIPTION = "Run script for testing framework."
    parser = SimulationArgumentParser(description=DESCRIPTION)
    input_args = parser.parse_sim_args()
    set_sim_and_run(input_args)
