from .WFARE import WFARE
from .FARE import FARE

class WFAREFiner(WFARE):

    def __init__(self, fare_model: FARE, model, policy_config, environment_config, mcts_config, batch_size=50,
                 training_buffer_size=200, validation_steps=10, expectation=None, sample_from_hard_examples=0) -> None:
        
        self.fare_model = fare_model

        super().__init__(model, policy_config, environment_config, mcts_config, batch_size, training_buffer_size, validation_steps, expectation, sample_from_hard_examples)


    def _compute_self_expectation(self, X, W_expectation, W_true):

        df, Y, trace, _, root_node = self.fare_model.predict(X, full_output=True, verbose=False)
        costs, Y = self.evaluate_trace_costs(trace, X, W_true)
        return df, costs[0], Y[0], trace, root_node