from ..models.WFARE import WFARE
from ..utils.functions import import_dyn_class, get_cost_from_tree, get_trace

from ..mcts.MCTSMul import MCTSMul

import pandas as pd
from tqdm import tqdm


class WFAREMul(WFARE):

    def __init__(self, model, policy_config, environment_config, mcts_config, batch_size=50, training_buffer_size=200, validation_steps=10, expectation=None, sample_from_hard_examples=0) -> None:
        super().__init__(model, policy_config, environment_config, mcts_config, batch_size, training_buffer_size, validation_steps, expectation, sample_from_hard_examples)
    
    def predict(self, X, W, G: dict=None, full_output :bool=False,
                verbose :bool=True, agent_only :bool=False,
                mcts_only :bool=False, skip_expectation_step:bool=False,
                mcts_steps: int=5, noise: float=0.2,
                previous_solutions: list=[],
                user_constraints: dict={},
                max_intervention_depth: int=None):
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
                **self.environment_config.get("additional_parameters"),
                user_constraints=user_constraints)
            
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

            # Override the intervention depth if needed
            if max_intervention_depth:
                env_validation.max_intervention_depth = max_intervention_depth

            mcts_validation = MCTSMul(
                env_validation, self.policy,
                minimum_cost = costs_exp if Y_exp > 0 else 10000,
                **self.mcts_config,
                previous_succesfull_solutions = previous_solutions
            )

            if mcts_only:
                mcts_validation.dir_epsilon = 1.0
            else:
                mcts_validation.exploration = True
                mcts_validation.dir_epsilon = noise
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