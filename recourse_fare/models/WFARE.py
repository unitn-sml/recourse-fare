from ..utils.functions import import_dyn_class, get_cost_from_tree, get_trace, compute_intervention_cost
from ..mcts import MCTSWeights
from ..agents.policy import Policy

from .FARE import FARE

from tensorboardX import SummaryWriter

import torch

from tqdm.auto import tqdm

import pandas as pd
import numpy as np


class WFARE(FARE):

    def __init__(self, model, policy_config, environment_config, mcts_config, 
                 batch_size=50, training_buffer_size=200, validation_steps=10,
                 expectation=None, sample_from_hard_examples=0.0) -> None:
        
        self.expectation = expectation
        self.sample_from_hard_examples = sample_from_hard_examples

        super().__init__(model, policy_config, environment_config, mcts_config, batch_size, training_buffer_size, validation_steps)

    def _init_policy(self, environment_config:dict, policy_config: dict):
        env = import_dyn_class(environment_config.get("class_name"))(None, None, None,
                                                                     **self.environment_config.get("additional_parameters"))

        num_programs = env.get_num_programs()
        additional_arguments_from_env = env.get_additional_parameters()

        # Set up the policy object
        self.policy = Policy(
            policy_config.get("observation_dim"),
            policy_config.get("encoding_dim"),
            policy_config.get("hidden_size"),
            num_programs,
            learning_rate=policy_config.get("learning_rate", 1e-3),
            **additional_arguments_from_env
        )

    def fit(self, X, W, max_iter=1000, X_hard=None, W_hard=None, verbose=True,
            tensorboard=None):

        # Initialize the various objects needed to train FARE
        self._init_training_objects()

        # If tensorboard
        if tensorboard:
            writer = SummaryWriter(tensorboard)

        # Dictionary containing the tensorboard values
        training_losses = {
            "actor": [],
            "arguments": [],
            "value": [],
            "total_nodes": []
        }

        with tqdm(range(1, max_iter+1), desc="Train FARE", disable=not verbose) as t:
            for iteration in t:

                t.set_description(
                    f"Train (Acc={float(self.training_statistics.print_statistics(string_out=True)):.3f}/Buff={self.buffer.get_memory_length()})"
                )

                # Extract both features and the corresponding weights
                features = X.sample(1) 
                weigths = W.iloc[[features.index[0]]]
                
                # If we have provide "hard" examples, sample from them with
                # a small probability. It should improve model performances.
                if W_hard is not None and X_hard is not None:
                    if np.random.rand() <= self.sample_from_hard_examples:
                        features = X_hard.sample(1) 
                        weigths = W_hard.iloc[[features.index[0]]]       
        
                # Compute the self-expectation
                costs_exp = 10000
                Y_exp = 0
                if self.expectation is not None:
                    _, costs_exp, Y_exp, _, _ = self._compute_self_expectation(
                        features.copy(), self.expectation, weigths 
                    )
                
                features = features.to_dict(orient='records')[0]
                weigths = weigths.to_dict(orient='records')[0]

                mcts = MCTSWeights(
                    import_dyn_class(self.environment_config.get("class_name"))(
                        features.copy(),
                        weigths.copy(),
                        self.model,
                        **self.environment_config.get("additional_parameters")
                        ), 
                    self.policy,
                    minimum_cost = costs_exp if Y_exp > 0 else 10000,
                    **self.mcts_config
                )

                traces, root_node, node_expanded = mcts.sample_intervention()   

                # Run one optimization step within the trainer
                act_loss, crit_loss, args_loss = self.trainer.train_one_step([traces])

                training_losses.get("actor").append(act_loss)
                training_losses.get("value").append(crit_loss)
                training_losses.get("arguments").append(args_loss)
                training_losses.get("total_nodes").append(node_expanded)

                # Perform the validation step
                if iteration % self.validation_steps == 0:

                    self.policy = self.trainer.policy

                    X_val = X.sample(10)
                    W_val = W.iloc[X_val.index]

                    _, validation_rewards, traces, costs, _ = self.predict(
                        X_val,
                        W_val,
                        full_output=True,
                        verbose=False)
                    lengths = [len(trace) for trace in traces]

                    # Update the statistics
                    self.training_statistics.update_statistics(validation_rewards, costs, lengths)

                    # If tensorboard, update tensorboard files
                    if tensorboard:
                        
                        # Add some information when training
                        writer.add_scalar("loss/actor", np.mean(training_losses.get("actor")), iteration)
                        writer.add_scalar("loss/value", np.mean(training_losses.get("value")), iteration)
                        writer.add_scalar("loss/arguments", np.mean(training_losses.get("arguments")), iteration)
                        writer.add_scalar("mcts/avg_node_expanded", np.mean(training_losses.get("total_nodes")), iteration)

                        # Add some information for the validation
                        avg_validity, avg_cost, avg_length = self.training_statistics.get_statistic()
                        writer.add_scalar('validation/avg_validity', avg_validity, iteration)
                        writer.add_scalar('validation/avg_cost', avg_cost, iteration)
                        writer.add_scalar('validation/avg_length', avg_length, iteration)

                    # Clean the logging
                    for k in training_losses.keys():
                        training_losses[k] = []
    
        # Copy the trainer policy to the object policy 
        self.policy = self.trainer.policy
    
    def predict(self, X, W, G: dict=None, full_output :bool=False,
                verbose :bool=True, agent_only :bool=False,
                mcts_only :bool=False, skip_expectation_step:bool=False,
                mcts_steps: int=5):
        """Generate counterfactual interventions given FARE.

        :param X: the dataset
        :param full_output: True if we want to return more than just the counterfactuals, defaults to False
        :type full_output: bool, optional
        :param verbose: if verbose, show a progress bar when performing inference, defaults to True
        :type verbose: bool, optional
        :param agent_only: run inference using only the trained agent, without MCTS, defaults to False
        :type agent_only: bool, optional
        :param mcts_only: run inferece using only the MCTS, without the trained agent, defaults to False
        :type mcts_only: bool, optional
        """

        X_dict = X.to_dict(orient='records')
        W_dict = W.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        root_nodes = []
        for i in tqdm(range(len(X)),  desc="Eval FARE", disable=not verbose):

            # Compute the self-expectation
            costs_exp = 10000
            Y_exp = 0
            if not skip_expectation_step and self.expectation is not None:
                df_exp, costs_exp, Y_exp, trace_exp, root_node_exp = self._compute_self_expectation(
                    X.iloc[[i]], self.expectation, W.iloc[[i]] 
                )

            env_validation = import_dyn_class(self.environment_config.get("class_name"))(
                X_dict[i].copy(),
                W_dict[i].copy(),
                self.model,
                **self.environment_config.get("additional_parameters"))
            
            # If we have the graph structure, override the preset one. 
            if G is not None:
                env_validation.structural_weights.set_scm_structure(G[i])

            # If we are using only the agent
            if agent_only:
                counterfactual, reward, trace, cost = self._predict_agent(env_validation)
                counterfactuals.append(counterfactual)
                Y.append(reward)
                traces.append(trace)
                costs.append(cost)
                root_nodes.append(None)
                continue

            mcts_validation = MCTSWeights(
                env_validation, self.policy,
                minimum_cost = costs_exp if Y_exp > 0 else 10000,
                **self.mcts_config
            )

            if mcts_only:
                mcts_validation.dir_epsilon = 1.0
            else:
                mcts_validation.exploration = False
                mcts_validation.number_of_simulations = mcts_steps

            # Sample an execution trace with mcts using policy as a prior
            trace, root_node, _ = mcts_validation.sample_intervention()
            task_reward = trace.task_reward

            cost, _ = get_cost_from_tree(root_node)

            # If we get a positive result with the expected value
            # at a lower cost, then we use it instead.
            if (costs_exp < cost and Y_exp == 1) or (Y_exp == 1 and task_reward <= 0):
                costs.append(costs_exp)
                traces += trace_exp
                root_nodes += root_node_exp
                Y.append(1 if Y_exp else 0)
                counterfactuals += df_exp.to_dict("records")
            else:
                costs.append(cost)
                traces.append(get_trace(env_validation, root_node))
                root_nodes.append(root_node)            
                Y.append(1 if task_reward > 0 else 0)
                counterfactuals.append(env_validation.features.copy())
        
        if full_output:
            return pd.DataFrame.from_records(counterfactuals), Y, traces, costs, root_nodes
        else:
            return pd.DataFrame.from_records(counterfactuals)
    
    def evaluate_trace_costs(self, traces: list, X, W, G: dict=None, **kwargs):

        X_dict = X.to_dict(orient='records')
        W_dict = W.to_dict(orient='records')

        costs = []
        recourse = []

        for idx, t in enumerate(traces):

            # Build the environment
            env = import_dyn_class(self.environment_config.get("class_name"))(
                X_dict[idx].copy(),
                W_dict[idx].copy(),
                self.model,
                **self.environment_config.get("additional_parameters"))
            
            # Set random type
            if G:
                env.structural_weights.set_scm_structure(G[idx])
            
            # Compute the intervention costs
            t_cost, has_recourse = compute_intervention_cost(
                env, X_dict[idx].copy(), t, custom_weights=W_dict[idx].copy(), **kwargs
            )
            costs.append(t_cost)
            recourse.append(has_recourse)
        
        return costs, recourse
    
    def _compute_self_expectation(self, X, W_expectation, W_true):

        df, Y, trace, cost, root_node = self.predict(X, W_expectation, full_output=True, verbose=False, skip_expectation_step=True)
        costs, Y = self.evaluate_trace_costs(trace, X, W_true)
        return df, costs[0], Y[0], trace, root_node