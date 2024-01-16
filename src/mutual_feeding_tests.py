import time
import os

from emukit.core.initial_designs.latin_design import LatinDesign

from coupledgp import (
    generate_data,
    load_data_from_file,
    CoupledGPModel,
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

# generate training data
if USE_DATA_PATH:
    X, Y = load_data_from_file(
        "./src/coupledgp/tests/x-sim-2.npy",
        "./src/coupledgp/tests/y-sim-2.npy",
    )
else:
    X, Y = generate_data(
        10, 100, "./src/coupledgp/tests/"
    )  # runs simulator for 100 days per sample (total 1000)
generate_finished = time.time()
eval_times["generate"] = (USE_DATA_PATH, generate_finished - start_time)

# training coupled model
model = CoupledGPModel(X, Y)
if USE_MODEL_PATH:
    model.load_models(
        "./src/coupledgp/tests/drift_model.npy",
        "./src/coupledgp/tests/population_model.npy",
    )
else:
    model.optimize()
    model.save_models(
        "./src/coupledgp/tests/drift_model_custom_kernel_2",
        "./src/coupledgp/tests/population_model_custom_kernel_2",
    )
model_finished = time.time()
eval_times["training"] = (USE_MODEL_PATH, model_finished - generate_finished)

print(model.compare_with_simulator())

print(
    model.compare_with_simulator(
        "./src/coupledgp/tests/mutation_rates.npy",
        "./src/coupledgp/tests/simulated_years_of_survival.npy",
    )
)
# quit()

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
