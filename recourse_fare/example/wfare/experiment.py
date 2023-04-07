from mpi4py import MPI
import pickle
import random

from sklearn.model_selection import train_test_split

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.models.InteractiveFARE import InteractiveFARE

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.user.user import NoiselessUser

import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser
import yaml

if __name__ == "__main__":

    # Add the argument parser
    parser = ArgumentParser()
    parser.add_argument("--questions", default=3, type=int, help="How many questions we shoudl ask.")

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

    # Read data and preprocess them
    X = pd.read_csv("recourse_fare/example/wfare/data.csv")[0:5000]
    y = X.income_target.apply(lambda x: 1 if x=="<=50K" else 0)
    X.drop(columns=["income_target", "predicted"], inplace=True)

    # We drop some columns we do not consider actionable. It makes the problem less interesting, but it does
    # show the point about how counterfactual interventions works. 
    #X.drop(columns=["fnlwgt", "age", "race", "sex", "native_country", "relationship", "education_num"], inplace=True)
    X.drop(columns=["fnlwgt", "age", "sex", "native_country", "relationship", "education_num"], inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Generate random weights. Weights needs to be non-null and positive
    single_weights = np.ones(15)
    W_train = [single_weights for _ in range(len(X_train))]
    W_test = np.abs(np.random.normal(loc=25, size=(len(X_test), 15)))

    # Build weights dataframes
    tmp_scm = AdultSCM(None)
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    # Build the dataframes with the weights
    W_train = pd.DataFrame(W_train, columns=keys_weights)
    W_test = pd.DataFrame(W_train, columns=keys_weights)

    # We are the master process
    if rank == 0:

        # Read the trained WFARE method from disk
        # The WFARE method contains the following:
        # - Blackbox classifier
        # - Custom preprocessor
        recourse_method = pickle.load(open("recourse.pth", "rb"))

        # Create the user model required
        user = NoiselessUser()

        # Create and interactive FARE object and predict the test instances
        interactive = InteractiveFARE(recourse_method, user, keys_weights, questions=int(args.questions), mcmc_steps=100, verbose=False)

        # Send the interactive FARE method to the childs
        bcast_data = interactive
    
    interactive = comm.bcast(bcast_data, root=0)

    # Given how many users we want to analyze, get a slice of
    # the data with the corresponding users
    iterations = 2
    perrank = iterations // size
    data_slice = (0 + rank * perrank,  0 + (rank + 1) * perrank)

    # Current slice
    X_test_slice, W_test_slice = X_test[data_slice[0]:data_slice[1]], W_test[data_slice[0]:data_slice[1]]

    # Generate the counterfactuals and traces
    (counterfactuals, Y, traces, costs, _), W_updated, failed_users = interactive.predict(X_test_slice, W_test_slice, full_output=True)

    # Send the complete results
    complete_trace = [counterfactuals, Y, traces, costs, W_updated, failed_users, data_slice[0], data_slice[1]]
    complete_trace = comm.gather(complete_trace, root=0)

    # If we are the master process, then we print all
    if rank == 0:
        
        # Sort the traces based on their interval
        complete_trace = sorted(complete_trace, key=lambda x: x[6])

        user_idx = []
        intervention_costs = []
        failed_users = []
        validity = []
        all_traces = []

        # Unwind the results and store the traces
        for counterfactuals, Y, traces, costs, W_updated, failed_users, start_slice, end_slice in complete_trace:
            
            user_range = list(range(start_slice, end_slice))

            user_idx += user_range
            validity += Y
            intervention_costs += costs
            failed_users += failed_users
            
            all_traces += [[idx, p,a] for t, idx in zip(traces, user_range) for p,a in t]

        # Save the validity, cost and elicitation result to disk
        data = pd.DataFrame(list(zip(user_idx, validity, intervention_costs, failed_users)), columns=["user_idx","recourse", "cost", "elicitation"])
        data.to_csv(f"validity_cost_elicitation-{args.questions}.csv", index=None)

        # Save the traces to disk
        data = pd.DataFrame(all_traces, columns=["user_idx", "action", "argument"])
        data.to_csv(f"traces-{args.questions}.csv", index=None)

        # Save estimated weights to disk
        weights = pd.concat([x[4] for x in complete_trace])
        weights.to_csv(f"estimated_weights-{args.questions}.csv", index=None)
