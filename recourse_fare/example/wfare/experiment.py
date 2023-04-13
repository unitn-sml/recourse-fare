from mpi4py import MPI
import dill as pickle
import random

from sklearn.model_selection import train_test_split

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.models.InteractiveFARE import InteractiveFARE

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.user.user import NoiselessUser, LogisticNoiseUser
from recourse_fare.utils.Mixture import MixtureModel

import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser

MIXTURE_MEAN_LIST =[
    [37, -25, 20, 5, -17, 42, -47, 2, 36, 46, -28, -33, 3, -3, 27],
    [-11, 19, 38, -47, 8, -22, 31, 21, 13, -2, -18, -21, -1, -1, -1],
    [-1, 3, -12, 20, -35, 38, 22, -3, 15, 21, -28, 3, -43, -5, -25],
    [-16, -1, 27, 41, 37, -5, -16, -16, -39, -49, -19, 19, 1, -31, -10],
    [19, -19, -7, -45, -40, -43, 33, 5, -44, 49, -22, -4, 28, 20, -13],
    [13, -25, -14, -10, 35, 32, -45, 43, 29, -24, 47, 16, 8, 22, 10]
]

WRONG_GRAPH_EDGES = [
    ("education", "workclass"),
    ("hours_per_week", "occupation"),
    ("workclass", "capital_gain"),
    ("capital_gain", "capital_loss")
]

if __name__ == "__main__":

    # Add the argument parser
    parser = ArgumentParser()
    parser.add_argument("--questions", default=3, type=int, help="How many questions we shoudl ask.")
    parser.add_argument("--test-set-size", default=100, type=int, help="How many users we should pick from the test set for evaluation.")
    parser.add_argument("--mcmc-steps", default=50, type=int, help="How many steps should the MCMC procedure perform.")
    parser.add_argument("--logistic-user", default=False, action="store_true", help="Use a logistic user rather than a noiseless one.")
    parser.add_argument("--wrong-graph", default=False, action="store_true", help="Use misspecified random graphs for the estimation phase.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Make the procedure verbose.")

    # Parse the arguments
    args = parser.parse_args()

    # Launch the script in a parallel fashion
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    bcast_data = None

    # Set seeds for reproducibility
    seed = 2023
    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read the trained WFARE method from disk
    # The WFARE method contains the following:
    # - Blackbox classifier
    # - Custom preprocessor
    recourse_method = pickle.load(open("recourse_fare/example/wfare/recourse.pth", "rb"))

    # Create the user model required
    if args.logistic_user:
        user = LogisticNoiseUser()
    else:
        user = NoiselessUser()

    # Get edges and nodes 
    tmp_scm = AdultSCM(None)
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    # Build the mixture (prior for the estimation)
    mixture = MixtureModel(
        mixture_means=MIXTURE_MEAN_LIST
    )

    # Create and interactive FARE object and predict the test instances
    interactive = InteractiveFARE(recourse_method, user, mixture, keys_weights,
                                  questions=int(args.questions), mcmc_steps=args.mcmc_steps,
                                  verbose=args.verbose)

    # Build the dataframes with the weights
    W_test = pd.read_csv("recourse_fare/example/wfare/weights_test.csv")
    W_test.rename(
        columns={c: eval(c) for c in W_test.columns},
        inplace=True
    )

    # Read data
    X = pd.read_csv("recourse_fare/example/wfare/test_data.csv")
    
    # Keep only the instances which are negatively classified
    X["predicted"] = recourse_method.model.predict(
        recourse_method.environment_config.get("additional_parameters").get("preprocessor").transform(X)
    )
    X = X[X.predicted == 1]
    X.drop(columns="predicted", inplace=True)
    X.reset_index(inplace=True, drop=True)

    # Given how many users we want to analyze, get a slice of
    # the data with the corresponding users
    iterations = args.test_set_size
    perrank = iterations // size
    data_slice = (0 + rank * perrank,  0 + (rank + 1) * perrank)

    # Generate graphs
    G = [{"edges": WRONG_GRAPH_EDGES} for i in range(len(X))]

    # Current slice
    X_test_slice, W_test_slice = X[data_slice[0]:data_slice[1]], W_test[data_slice[0]:data_slice[1]]
    G_test_slice = G[data_slice[0]:data_slice[1]]

    # Generate the counterfactuals and traces
    (counterfactuals, Y, traces, costs_e, _), W_updated, failed_users = interactive.predict(
        X_test_slice, W_test_slice, G_test_slice,
        full_output=True, use_true_graph=not args.wrong_graph)

    # Regenerate the true costs, given the found traces
    costs = interactive.evaluate_trace_costs(traces, X_test_slice, W_test_slice, G_test_slice, use_true_graph=True)

    # Send the complete results
    complete_trace = [counterfactuals, Y, traces, costs, W_updated, failed_users, data_slice[0], data_slice[1]]
    complete_trace = comm.gather(complete_trace, root=0)

    # If we are the master process, then we print all
    if rank == 0:
        
        # Sort the traces based on their interval
        complete_trace = sorted(complete_trace, key=lambda x: x[6])

        user_idx = []
        intervention_costs = []
        failed_users_all = []
        validity = []
        all_traces = []

        # Unwind the results and store the traces
        for counterfactuals, Y, traces, costs, W_updated, failed_users, start_slice, end_slice in complete_trace:
            
            user_range = list(range(start_slice, end_slice))

            user_idx += user_range
            validity += Y
            intervention_costs += costs
            failed_users_all += failed_users
            
            all_traces += [[idx, p,a] for t, idx in zip(traces, user_range) for p,a in t]

        # Save the validity, cost and elicitation result to disk
        data = pd.DataFrame(list(zip(user_idx, validity, intervention_costs, failed_users_all)), columns=["user_idx","recourse", "cost", "elicitation"])
        data.to_csv(f"validity_cost_elicitation-{args.questions}-{args.wrong_graph}-{args.logistic_user}.csv", index=None)

        # Save the traces to disk
        data = pd.DataFrame(all_traces, columns=["user_idx", "action", "argument"])
        data.to_csv(f"traces-{args.questions}-{args.wrong_graph}-{args.logistic_user}.csv", index=None)

        # Save estimated weights to disk
        weights = pd.concat([x[4] for x in complete_trace])
        weights.to_csv(f"estimated_weights-{args.questions}-{args.wrong_graph}-{args.logistic_user}.csv", index=None)
