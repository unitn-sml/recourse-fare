from tqdm import tqdm

from recourse_fare.models import WFARE
from recourse_fare.user.user import User
from recourse_fare.utils.Mixture import MixtureModel

from ..utils.functions import run_automaton, compute_intervention_cost
from ..utils.functions import import_dyn_class, backtrack_eus
from ..environment_w import EnvironmentWeights
from ..mcts import MCTSWeights, MCTS
from . import WFARE, WFAREFiner, WEFARE
from .PEAR import PEAR
from ..user.user import User, NoiselessUser, LogisticNoiseUser
from ..user.choice import ChoiceGenerator, SliceSamplerNoiseless, SliceSamplerLogistic
from ..utils.Mixture import MixtureModel

import numpy as np
import pandas as pd

def convert_candidate_str(candidate_intervention):
   return "/".join(["-".join([str(y) for y in x]) for x in candidate_intervention])

class XPEAR(PEAR):

    def __init__(self, recourse_model: WEFARE, user: User, mixture: MixtureModel, features: list,
                 questions: int = 10, choice_set_size: int = 2, mcmc_steps=100, n_particles=100, 
                 verbose: bool = False, random_choice_set: bool = False) -> None:
        
        assert isinstance(recourse_model, WEFARE), "XPEAR accept only a trained WEFARE object!"
        
        super().__init__(recourse_model, user, mixture, features, questions, 
                         choice_set_size, mcmc_steps, n_particles, verbose, random_choice_set)
    
    def _generate_potential_set(self, env: EnvironmentWeights, questions_to_avoid: list=[]):

        current_environment_state = env.features.copy()
        action_choices = env.get_current_actions()

        # Potential action we can ask to the user
        potential_set = []

        current_interventions_random = set()

        for program, argument, program_index, argument_index in tqdm(action_choices, disable=not self.verbose):

            # Avoid asking always the same question at the first iteration.
            if (program, argument, current_environment_state) in questions_to_avoid:
                continue

            # Set the best candidate
            best_candidate_cost = np.inf
            best_candidate = None

            # Compute candidate intervention using the estimated weights so far
            # Perform both random walk over the efare automaton and the pick the best
            # solution with the smallest cost so far. This is done for all the potential
            # actions.
            for randomize_opt in [True, False]:
                
                best_cost, best_intervention, \
                    _, _ = run_automaton(self.recourse_model,
                                        env,
                                        current_environment_state.copy(),
                                        deterministic_actions=[(program, argument)],
                                        randomize=randomize_opt)

                # If we find a good trace, then we add this action to the potential set we will show to the user
                if best_intervention:

                    candidate_intervention = best_intervention

                    # Avoid processing empty interventions
                    if len(candidate_intervention) == 0:
                        continue
                    
                    # Check if we found this candidate intervention already.
                    # The idea is to include random walks on the graph as well as correct solutions.
                    if convert_candidate_str(candidate_intervention) in current_interventions_random:
                        continue
                    else:
                        current_interventions_random.add(convert_candidate_str(candidate_intervention))

                    # Compute the cost of this intervention and save the best ones
                    candidate_cost, _ = compute_intervention_cost(
                        env,
                        current_environment_state.copy(),
                        candidate_intervention
                    )
                    
                    assert candidate_cost == best_cost, (candidate_cost, best_cost)

                    # Replace the best_candidate
                    if candidate_cost < best_candidate_cost:
                        best_candidate = candidate_intervention.copy()

            # If we found a solution, then add it here
            if best_candidate:
                potential_set.append(
                    [program, argument, best_candidate,
                     current_environment_state.copy(),
                     env.features.copy()]
                )
            
            env.features = current_environment_state.copy()

        # Shuffle the potential set
        np.random.shuffle(potential_set)

        if self.verbose:
            print("Potential set size: ", len(potential_set))
        
        return potential_set

