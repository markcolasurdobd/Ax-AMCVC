from typing import Sequence
from ax.api.client import Client
from ax.core import MultiObjectiveOptimizationConfig
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.generation_node import GenerationNode
from ax.adapter.registry import Generators
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig
from ax.generation_strategy.generation_strategy import GenerationStrategy
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from ax.utils.stats.model_fit_stats import MSE
import pandas as pd

# Instantiate client
client = Client()

# Initialize composition variables
dio = RangeParameterConfig(name='dioxolane', parameter_type='float', bounds=(0, 100))
met = RangeParameterConfig(name='methanol', parameter_type='float', bounds=(0, 100))
total_ch = RangeParameterConfig(name='total_ch', parameter_type='float', bounds=(0, 10))
cha = RangeParameterConfig(name='cha', parameter_type='float', bounds=(0, 100))
ssd = RangeParameterConfig(name='ssd', parameter_type='float', bounds=(0, 5))

# Read and format pre-existing data
data_path = r"./amcvc_data.csv"
df = pd.read_csv(data_path)
df = df.astype(float)   # Convert whole df to float
X = df[['dioxolane', 'methanol', 'total_ch', 'cha', 'ssd']]  # Independent variables
y1 = df['chx_d4']
y2 = df['ssd_d4']
y = pd.concat([y1, y2], axis = 1)
preexisting_trials = [tuple([{column:X.loc[formulation, column] for column in X.columns}, {column:y.loc[formulation, column] for column in y.columns}]) for formulation in df.index]

# Configure experiment with dependent variables
client.configure_experiment(parameters=[dio, met, total_ch, cha, ssd],
                            parameter_constraints=['dioxolane + methanol <= 100.0'])

# Configure optimization
client.configure_optimization(objective="chx_d4, ssd_d4")

# Configure custom generation strategy
surrogate_spec = SurrogateSpec(
    model_configs=[
        ModelConfig(
            botorch_model_class=SingleTaskGP,
            covar_module_class=MaternKernel,
            covar_module_options={"nu": 2.5},
        ),
    ],
    eval_criterion=MSE,  # Select the model to use as the one that minimizes mean squared error.
    allow_batched_models=False,  # Forces each metric to be modeled with an independent BoTorch model.
    # If we wanted to specify different options for different metrics.
    # metric_to_model_configs: dict[str, list[ModelConfig]]
)

generator_spec = GeneratorSpec(
    generator_enum=Generators.BOTORCH_MODULAR,
    model_kwargs={
        "surrogate_spec": surrogate_spec,
        "botorch_acqf_class": qLogNoisyExpectedImprovement,
        # Can be used for additional inputs that are not constructed
        # by default in Ax. We will demonstrate below.
    },
    # We can specify various options for the optimizer here.
    model_gen_kwargs = {
        "model_gen_options": {
            "optimizer_kwargs": {
                "num_restarts": 20,
                "sequential": False,
                "options": {
                    "batch_limit": 5,
                    "maxiter": 200,
                },
            },
        },
    }
)
botorch_node = GenerationNode(
    node_name = "LogEI",
    generator_specs=[generator_spec]
)
gs = GenerationStrategy(
    name = "LogEI Generation Strategy",
    nodes = [botorch_node]
)
client.set_generation_strategy(generation_strategy = gs)

# Attach preexisting data to experiment
for parameters, data in preexisting_trials:
    # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
    trial_index = client.attach_trial(parameters=parameters)
    client.complete_trial(trial_index=trial_index, raw_data=data)

# Generate next trials
trials = client.get_next_trials(max_trials=3)

# Compute sum to make sure material components sum to 100
# for trial in trials.keys():
#     comp_sum = 0
#     trials[trial]['oil'] = 0
#     for comp in trials[trial].keys():
#         comp_sum += trials[trial][comp]
#     trials[trial]['oil'] = 100 - comp_sum

# Display trials
trials

# Save experiment
# =============================================================================
# SAVE_PATH = r"C:\Users\10354191\OneDrive - BD\Projects\MDS\TPE Stopper\Code\Optimization.json"
# client.save_to_json_file(SAVE_PATH)
# =============================================================================
