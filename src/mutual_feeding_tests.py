import time
import os

from emukit.core.initial_designs.latin_design import LatinDesign

from coupledgp import (
    coupled_space,
    generate_data,
    load_data_from_file,
    CoupledGPModel,
    NUM_TRAITS,
)

base_path = os.path.dirname(__file__)
USE_DATA_PATH = True
USE_MODEL_PATH = False

TEST_PREDICTION = True
VISUALIZE_RESULTS = True
TEST_SENSITIVITY = True

SHOW_GRAPHS = False

# timing initialization
start_time = time.time()
eval_times = dict()

# training data
# for n, n_s in zip([1, 2, 5, 10, 500], [500, 250, 100, 50, 1]):
#     print(f"Generating data for {n} runs of {n_s} steps:")
#     X, Y = generate_data(n, n_s, "./src/coupledgp/tests/")
# for n, n_s in zip([1, 2, 5, 10, 500], [500, 250, 100, 50, 1]):
#     print(f"Optimizing model for {n} runs of {n_s} steps:")
#     X, Y = load_data_from_file(
#         f"./src/coupledgp/tests/x-{n}-simfor-{n_s}-{NUM_TRAITS}-traits.npy",
#         f"./src/coupledgp/tests/y-{n}-simfor-{n_s}-{NUM_TRAITS}-traits.npy",
#     )
#     model = CoupledGPModel(X, Y)
#     model.optimize()
#     model.save_models(
#         f"./src/coupledgp/tests/drift_model_{n}_runs_{n_s}_steps",
#         f"./src/coupledgp/tests/population_model_{n}_runs_{n_s}_steps",
#     )

# quit()

# generate training data
if USE_DATA_PATH:
    X, Y = load_data_from_file(
        "./src/coupledgp/tests/x-1-simfor-500-8-traits.npy",
        "./src/coupledgp/tests/y-1-simfor-500-8-traits.npy",
    )
else:
    X, Y = generate_data(
        2, 100, "./src/coupledgp/tests/"
    )  # runs simulator for 100 days per sample
generate_finished = time.time()
eval_times["generate"] = (USE_DATA_PATH, generate_finished - start_time)

# training coupled model
model = CoupledGPModel(X, Y)
# if USE_MODEL_PATH:
#     model.load_models(
#         "./src/coupledgp/tests/drift_model.npy",
#         "./src/coupledgp/tests/population_model.npy",
#     )
# else:
#     model.optimize()
#     model.save_models(
#         "./src/coupledgp/tests/drift_model_custom_kernel_2",
#         "./src/coupledgp/tests/population_model_custom_kernel_2",
#     )
model_finished = time.time()
eval_times["training"] = (USE_MODEL_PATH, model_finished - generate_finished)

print(
    model.compare_with_simulator(
        "./src/coupledgp/tests/mutation_rates.npy",
        "./src/coupledgp/tests/simulated_years_of_survival.npy",
    )
)
quit()

# predicting with coupled model
if TEST_PREDICTION:
    model.plot_coupled_model(1000, show_plot=SHOW_GRAPHS)
prediction_finished = time.time()
eval_times["prediction"] = (
    TEST_PREDICTION,
    prediction_finished - model_finished,
)

print("Evaluation times:")
for k in eval_times:
    print(
        f"{k} (shortcutted: {eval_times[k][0]}, graphs: {SHOW_GRAPHS}): {eval_times[k][1]}s"
    )

quit()
# visualizing model results
if VISUALIZE_RESULTS:
    model.plot_drift_model(show_plot=SHOW_GRAPHS)
    model.plot_population_model(show_plot=SHOW_GRAPHS)
plotting_finished = time.time()
eval_times["plotting"] = (
    VISUALIZE_RESULTS,
    plotting_finished - prediction_finished,
)

# sensitivity analysis
if TEST_SENSITIVITY:
    model.drift_sensitivity_analysis(show_plot=SHOW_GRAPHS)
    model.population_sensitivity_analysis(show_plot=SHOW_GRAPHS)
sensitivity_finished = time.time()
eval_times["sensitivity"] = (
    TEST_SENSITIVITY,
    sensitivity_finished - plotting_finished,
)
