import torch
import importlib

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from itertools import permutations

from ..environment_w import EnvironmentWeights

def compute_intervention_cost(env: EnvironmentWeights, env_state: dict, intervention: list,
                                  custom_weights: dict=None, bin_argument: bool=True) -> float:
        """
        Compute the cost of an intervention. It does not modify the environment, however, this
        function is unsafe to be used in a multi-threaded context. Unless the object in replicated
        in each process separately
        :param intervention: ordered list of action/value/type tuples
        :param custom_env: feature updates which are applied before computing the cost (not persistent)
        :param estimated: if True, we compute the cost using the estimated graph
        :param bin_argument: if True, convert the argument to a binned value.
        :return: intervention cost
        """

        prev_state = env.features.copy()
        prev_weights = env.weights.copy()

        env.features = env_state.copy()
        env.weights = custom_weights if custom_weights else env.weights

        intervention_cost = 0
        for action, value in intervention:
            prog_idx = env.prog_to_idx.get(action)
            if bin_argument:
                value_idx = env.inverse_complete_arguments.get(value).get(action)
                assert value_idx is not None, (action, value)
                intervention_cost += env.get_cost(prog_idx, value_idx)
            else:
                intervention_cost += env.get_cost_raw(prog_idx, value)
            
            # Perform the action on the environment
            # We avoid using act() because it is very costly (since it generates an observation).
            env.has_been_reset = True
            assert action in env.primary_actions, 'action {} is not defined'.format(action)
            env.prog_to_func[action](value)
        
        recourse = env.prog_to_postcondition(prev_state.copy(), env.features.copy())

        env.features = prev_state
        env.weights = prev_weights

        return intervention_cost, recourse

def import_dyn_class(path: str):
    """
    Import dynamically a python class which can be then used
    :param path: path to the class
    :return: the class which can be instantiated
    """

    class_name = path.split(".")[-1]
    module_name = ".".join(path.split(".")[:-1])

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_

def fix_policy(policy):
    """
    This fix potential issues which happens withing the policy. Namely, NaN probabilities
    or negative probabilites (which should not happen anyway).
    :param policy: the policy we need to fix
    :return: a safe version of the policy
    """

    safe_policy = torch.where(torch.isnan(policy), torch.zeros_like(policy), policy)
    safe_policy = torch.max(safe_policy, torch.tensor([0.]))

    # This means that we changed something. Therefore we need to recompute
    # the policy taking into account the corrected values
    if safe_policy.sum() != 1.0 and safe_policy.sum() != 0.0:
        safe_policy = safe_policy / safe_policy.sum()

    return safe_policy

def compute_q_value(node, qvalue_temperature):
    if node.visit_count > 0.0:
        values = torch.FloatTensor(node.total_action_value)
        softmax = torch.exp(qvalue_temperature * values)
        softmax = softmax / softmax.sum()
        q_val_action = float(torch.dot(softmax, values))
    else:
        q_val_action = 0.0
    return q_val_action

def get_cost_from_env(env, action, args, env_state = None):

    tmp_state = None
    if env_state is not None:
        tmp_state = env.get_state()
        env.reset_to_state(env_state)
        
    action_index = env.prog_to_idx.get(action)
    args_index = env.inverse_complete_arguments.get(args).get(action)

    cost = env.get_cost(action_index, args_index)

    if tmp_state:
        env.reset_to_state(tmp_state)

    return cost

def get_single_action_costs(root_node):

    costs = []
    stack = [root_node]
    while stack:

        assert len(stack) == 1

        cur_node = stack.pop(0)

        if cur_node.single_action_cost:
            costs.append(cur_node.single_action_cost)

        for child in cur_node.childs:
            if child.selected:
                stack.append(child)
    return costs

def get_cost_from_tree(root_node):

    cost = 0
    stack = [root_node]
    length = 0
    while stack:

        assert len(stack) == 1

        cur_node = stack.pop(0)
        
        length += 1
        cost = cur_node.cost

        for child in cur_node.childs:
            if child.selected:
                stack.append(child)
    return cost, length

def get_trace(env, root_node):

    trace = []
    stack = [root_node]
    while stack:
        cur_node = stack[0]

        if cur_node.selected:
            if cur_node.program_from_parent_index is not None:

                action_name = env.get_program_from_index(cur_node.program_from_parent_index)
                trace.append([action_name, cur_node.args])

        stack = stack[1:]
        for child in cur_node.childs:
            stack.append(child)
    return trace

def backtrack_eus(env, potential_set, max_choice_set, choice_generator, user, sampler, asked_questions,
                  choices_id, choices, random_choice_set=False, verbose=False):

    # If random, then pick up three actions which we did not see before:
    if random_choice_set: 

        if verbose:
            print("[*] Choosing random choice set.")
        
        # Add additional fallback
        if len(potential_set) < max_choice_set:
            return None

        already_asked = True
        counter = 100
        while(already_asked and counter > 0):
            already_asked = False
            ids = np.random.choice(list(range(len(potential_set))), max_choice_set, replace=False)
            choices = [potential_set[x] for x in ids]
            for (c_env, q) in asked_questions:
                current_c = set([(c[0], c[1], tuple(c[2])) for c in choices])
                if set(q) == current_c and c_env == env.features:
                    already_asked = True
                    counter -= 1
                    break
        
        return choices

    if len(choices) >= max_choice_set or len(potential_set) == len(choices_id):

        # Check if we already asked this set
        already_asked = False
        for (c_env, q) in asked_questions:
            current_c = set([(c[0], c[1], tuple(c[2])) for c in choices])
            if set(q) == current_c and c_env == env.features:
                already_asked = True
                break

        # If we did, we just return None and we continue the cycle.
        # If it is okay, we just return the choices done so far
        if already_asked:
            return None
        else:
            return choices

    else:

        # Compute eus values so far
        eus_values = []

        # Compute current eus value
        current_eus_value = choice_generator.compute_eus(env, user, sampler.get_current_particles(), choices)

        for id, x in enumerate(potential_set):

            # Skip actions we already saw
            if id in choices_id:
                continue

            program = x[0]
            value = x[1]
            intervention = x[2]
            prev_memory_status = x[3]
            prev_initial_memory = x[4]

            # Compute eus value
            eus = choice_generator.compute_eus(env, user, sampler.get_current_particles(), [x] + choices)
            eus_values.append(
                [id, program, value, eus-current_eus_value, intervention, prev_memory_status.copy(), prev_initial_memory.copy()])

        eus_values = sorted(eus_values, key=lambda x: x[3], reverse=True)

        resulting_choice = None
        if len(eus_values) > 0:

            for idx in range(len(eus_values)):

                best_choice_atm = eus_values[idx][1:]
                best_choice_atm.pop(2)

                choices.append(best_choice_atm)
                choices_id.append(eus_values[idx][0])

                # Recursive call with the new choice set
                resulting_choice = backtrack_eus(env, potential_set, max_choice_set, choice_generator, user, sampler, asked_questions, choices_id,
                          choices)

                # Pop previous choice and redo everything if this didn't work
                if resulting_choice is None:
                    choices.pop()
                    choices_id.pop()
                else:
                    break

        return resulting_choice

def plot_sampler(chain, dim, sampler=None, start=0):
    plt.figure(figsize=(16,1.5*(dim-start)))
    for n in range(dim-start):
        plt.subplot2grid((dim-start, 1), (n, 0))
        plt.plot(chain[:,:,n+start], alpha=0.5)
    plt.tight_layout()
    plt.show()

def compute_average_regret(full_potential_actions, choice_set, particles, user):

    # 0) Save weights to compute them again later
    pw_weights, pw_nodes  = user.features.estimated_graph.get_weights()

    # 1) Compute intervention with the best EU
    best_eu = []
    for _, _, intervention, choice_env, _ in full_potential_actions:
        current_cost = 0
        for w in particles:
            w = {k: v for k, v in zip(pw_nodes, w)}
            user.features.estimated_graph.update_weights(w)
        
            int_cost_tmp = user.features.compute_intervention_cost(intervention,
                                                                    custom_env=choice_env.copy(),
                                                                    estimated=True)
            current_cost += int_cost_tmp 
        best_eu.append((np.mean(current_cost), intervention, choice_env.copy()))

    # 1.2) Sort the EU and pick the optimal intervention
    best_eu = sorted(best_eu, key=lambda x: x[0])
    best_eu_intervention = best_eu[0][1]
    best_eu_state = best_eu[0][2]

    differences = []
    # 2) For each weight, compute the difference between the optimal cost
    for w in particles:
        
        w = {k: v for k, v in zip(pw_nodes, w)}
        user.features.estimated_graph.update_weights(w)
        
        intervention_costs = []
        for _, _, intervention, choice_env, _ in choice_set:
            int_cost_tmp = user.features.compute_intervention_cost(intervention,
                                                                    custom_env=choice_env.copy(),
                                                                    estimated=True)
            intervention_costs.append((int_cost_tmp, intervention))
        
        # Get the max_O C(I|w)
        intervention_costs = sorted(intervention_costs, key=lambda x: x[0])
        best_cs_eu_cost = intervention_costs[0][0]

        # Compute the cost of the intervention maximizing the EU with the current weight
        best_eu_cost = user.features.compute_intervention_cost(best_eu_intervention, custom_env=best_eu_state.copy(), estimated=True)
        
        # Compute the difference
        differences.append(
            best_cs_eu_cost-best_eu_cost
        )
    
    regret = sum(differences)/len(differences)

    # Restore the previous weights
    w = {k: v for k, v in zip(pw_nodes, pw_weights)}
    user.features.estimated_graph.update_weights(w)

    return regret

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False