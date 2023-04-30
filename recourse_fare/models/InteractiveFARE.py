from tqdm import tqdm
from ..utils.functions import import_dyn_class, backtrack_eus
from ..environment_w import EnvironmentWeights
from ..mcts import MCTSWeights
from ..models import WFARE
from ..user.user import User, NoiselessUser, LogisticNoiseUser
from ..user.choice import ChoiceGenerator, SliceSamplerNoiseless, SliceSamplerLogistic
from ..utils.Mixture import MixtureModel

import numpy as np
import pandas as pd

class InteractiveFARE:

    def __init__(self, recourse_model: WFARE, user: User, mixture: MixtureModel, features: list,
                 questions: int=10, choice_set_size: int=2,
                 mcmc_steps=100, n_particles=100, verbose: bool=False,
                 random_choice_set: bool=False) -> None:
        self.questions = questions
        self.choice_set_size = choice_set_size

        self.recourse_model: WFARE = recourse_model
        self.model = recourse_model.model

        self.verbose = verbose

        self.environment_config = self.recourse_model.environment_config
        self.mcts_config = self.recourse_model.mcts_config

        self.mixture = mixture

        self.random_choice_set = random_choice_set

        if isinstance(user, LogisticNoiseUser):
            # In the case of the logistic user, the sampler will use the noiseless_user
            # only to compute the intervention costs, which are independent of the
            # the response model. 
            sampler_type = SliceSamplerLogistic
        else:
            sampler_type = SliceSamplerNoiseless

        self.sampler = sampler_type(nodes=features, 
                                    nsteps=mcmc_steps,
                                    mixture=self.mixture,
                                    nparticles=n_particles,
                                    verbose=verbose)
        self.choice_generator = ChoiceGenerator()

        self.user: User = user
        self.noiseless_user: NoiselessUser = NoiselessUser()

    def predict(self, X, W, G: dict=None, full_output=False, batching: int=1, **kwargs):

        X_dict = X.to_dict(orient='records')
        W_dict = W.to_dict(orient='records')

        # New weights and failed users
        W_updated = []
        failed_user_estimation = []

        for i in tqdm(range(len(X)), desc="Predict: "):

            # Reset constraints for the new user
            self.sampler.reset_constraints()

            # Flag to signal we failed to estimate the weights for a user
            failed_user = False
            no_candidate_at_first_question = False
            no_candidate_intervention_found = False
            some_questions_asked = False

            # Compute the expected value of the mixture
            expected_value_mixture = np.mean(self.mixture.sample(100), axis=0)

            # Weights we are going to estimate
            estimated_weights = {k:v for k,v in zip(W_dict[i].keys(), expected_value_mixture)}
            
            # Build the environment
            env: EnvironmentWeights = import_dyn_class(self.environment_config.get("class_name"))(
                X_dict[i].copy(),
                estimated_weights,
                self.model,
                **self.environment_config.get("additional_parameters"),
            )
            
            # Store the previously asked actions to avoid asking them again.
            asked_actions = []

            # Here we save those actions which enable recourse
            # at some point in the interaction with the user.
            questions_to_avoid_because_succesfull = []

            # We start asking N questions to the user.
            for question in tqdm(range(self.questions), desc=f"Eliciting user {i}: "):

                # If we reached recourse, we need to reset the environment
                env.start_task()
                if env.get_reward() > 0 or failed_user or no_candidate_at_first_question or no_candidate_intervention_found or some_questions_asked:
                
                    env.features = X_dict[i].copy()
                    env.weights = estimated_weights
                    no_candidate_at_first_question = False
                    no_candidate_intervention_found = False
                    some_questions_asked = False
                    failed_user = False

                current_env_state = env.features.copy()
                
                # Generate the potential set of actions. We avoid asking the same question, if it
                # reached recourse after.
                potential_set = self._generate_potential_set(env, questions_to_avoid_because_succesfull)
                assert env.features == current_env_state

                # Given the potential set of actions, pick the choice set maximizing EUS
                choices = self._generate_choice_set(env, potential_set, asked_actions)
                assert env.features == current_env_state

                # Assert that everything is going well
                (can_continue,
                 no_candidate_at_first_question,
                 no_candidate_intervention_found,
                 some_questions_asked) = self._assert_elicitation_state(choices, question)

                if can_continue:

                    # Set the correct user graph, if it is specified
                    # Set the corresponding graph if it exists
                    if G is not None:
                        env.structural_weights.set_scm_structure(G[i])

                    # Show the choices to the user and let her pick the best one
                    # We supply the user the real weights.
                    (best_action, best_value, best_intervention, best_previous_state, _), _ = self._ask_user(
                        env, choices, W_dict[i].copy()
                    )
                    assert env.features == current_env_state

                    # Reset the graph to the default configuration
                    env.structural_weights.reset_scm_to_default()

                    # Sample the weights given the current user answers
                    # The sampling is done with the estimated weights
                    if question % batching == 0: 
                        try:
                            splr, _ = self.sampler.sample([((best_action, best_value, best_intervention.copy()), choices)], env)

                            # If we did not find enough particles, then we abort
                            # and we ask a new question to the user.
                            if len(self.sampler.current_particles) == 0:
                                print(f"No particle found (user {i}, question {question}).")
                                failed_user = True
                                break

                        except Exception as e:
                            # If we fail sampling, then we skip this user and we go to the next
                            print("Exception while sampling: ", e)
                            failed_user = True
                            break
                        assert env.features == current_env_state

                    # Append current choice set, so that we avoid to build it in the future
                    # We also record the environment where we asked the question, such to avoid
                    # asking the same question on the same env.
                    asked_actions.append((
                            env.features.copy(),
                            [(c[0], c[1], tuple(c[2])) for c in choices]
                        )
                    )

                    # Get the new weights and update our estimate
                    # We can do it since when we compute the new weights, is basically the mean over
                    # all the particles
                    new_estimated_weights = self.sampler.get_mean_high_likelihood_particles()
                    estimated_weights = new_estimated_weights if new_estimated_weights else estimated_weights
                    failed_user = True if new_estimated_weights is None else failed_user

                    # Apply the action and redo the cycle
                    env.has_been_reset = True
                    env.act(best_action, best_value)
                    assert env.features != current_env_state

                    # Append best action to the deterministic set
                    if env.get_reward() > 0:
                        questions_to_avoid_because_succesfull.append(
                            (best_action, best_value, best_previous_state.copy())
                        )

            # Append the result if needed
            failed_user_estimation.append(1 if failed_user else 0)
            
            # We append the estimated weights for this user
            W_updated.append(estimated_weights.copy())
        
        # Once we have the estimated weights, we predict the counterfactuals using the
        # weight aware model.
        if full_output:
            return self.recourse_model.predict(
                X, pd.DataFrame.from_records(W_updated), None,
                full_output=full_output, **kwargs
            ), pd.DataFrame.from_records(W_updated), failed_user_estimation
        else:
            return self.recourse_model.predict(
                X, pd.DataFrame.from_records(W_updated), None,
                **kwargs
            )

    def _assert_elicitation_state(self, choice_set, question):

        can_continue = False
        no_candidate_at_first_question = False
        some_questions_asked = False
        no_candidate_intervention_found = False

        if len(choice_set) == 0 and question == 0:
            print("No candidate intervention found (no choices)")
            no_candidate_at_first_question = True
        elif len(choice_set) == 0 and question != 0:
            print("No candidate intervention found (asked some questions)")
            some_questions_asked = True
        elif len(choice_set) == 0:
            print("No candidate intervention found (with choices)")
            no_candidate_intervention_found = True
        else:
            can_continue = True
        
        return can_continue, no_candidate_at_first_question, some_questions_asked, no_candidate_intervention_found

    def _ask_user(self, env: EnvironmentWeights, choices: list, custom_weights: dict=None):
        return self.user.compute_best_action(env, choices, custom_weights)       

    def _generate_choice_set(self, current_features: dict, potential_set: list, asked_actions: list):
        
        if self.verbose:
            print("[*] Generating the choice set.")

        # If we did not find enough interventions, we return an empty choice set
        if len(potential_set) < self.choice_set_size:
            return []
    
        choices = backtrack_eus(current_features, potential_set, self.choice_set_size, 
                                self.choice_generator, self.noiseless_user, self.sampler,
                                asked_actions, [], [], random_choice_set=self.random_choice_set,
                                verbose=self.verbose)
        
        # We did not find enough combinations to make a choice set
        # therefore avoid asking this question.
        if len(choices) < self.choice_set_size:
            choices = []
        
        if self.verbose:
            print("[*] Done generating the choice set.")

        return choices
    
    def _generate_potential_set(self, env: EnvironmentWeights, questions_to_avoid: list=[]):

        current_environment_state = env.features.copy()
        action_choices = env.get_current_actions()

        # Potential action we can ask to the user
        potential_set = []

        for program, argument, program_index, argument_index in tqdm(action_choices, disable=not self.verbose):

            # Avoid asking always the same question at the first iteration.
            if (program, argument, current_environment_state) in questions_to_avoid:
                continue
            
            mcts_validation = MCTSWeights(
                env, self.recourse_model.policy,
                **self.mcts_config
            )
            mcts_validation.exploration = True
            mcts_validation.number_of_simulations = 5

            trace, _, _ = mcts_validation.sample_intervention(deterministic_actions=[[program_index, argument_index]])

            if trace.task_reward > 0:
                actions_found = [env.get_program_from_index(idx) for idx
                                in trace.previous_actions[1:]]
                args_found = trace.program_arguments[1:]

                candidate_intervention = list(zip(actions_found, args_found))

                potential_set.append(
                    [program, argument, candidate_intervention,
                    current_environment_state.copy(),
                    env.features.copy()
                    ]
                )
            
            env.features = current_environment_state.copy()

        # Shuffle the potential set
        np.random.shuffle(potential_set)

        if self.verbose:
            print("Potential set size: ", len(potential_set))
        
        return potential_set

