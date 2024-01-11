import time
import os

from emukit.core.initial_designs.latin_design import LatinDesign

from coupledgp import (
    generate_data,
    load_data_from_file,
    coupled_space,
    CoupledGPModel,
)

base_path = os.path.dirname(__file__)
USE_DATA_PATH = False
USE_MODEL_PATH = False

TEST_PREDICTION = False
VISUALIZE_RESULTS = False
TEST_SENSITIVITY = True

# timing initialization
start_time = time.time()
eval_times = dict()

# generate training data
if USE_DATA_PATH:
    X, Y = load_data_from_file(
        "./src/coupledgp/tests/x-100000.npy",
        "./src/coupledgp/tests/y-100000.npy",
    )
else:
    X, Y = generate_data(100000, "./src/coupledgp/tests/")
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
        "./src/coupledgp/tests/drift_model",
        "./src/coupledgp/tests/population_model",
    )
model_finished = time.time()
eval_times["training"] = (USE_MODEL_PATH, model_finished - generate_finished)

# predicting with coupled model
if TEST_PREDICTION:
    design = LatinDesign(coupled_space)
    test_inputs = design.get_samples(5)
    test_outputs = model.predict(test_inputs)
    print(test_outputs)
prediction_finished = time.time()
eval_times["prediction"] = (
    TEST_PREDICTION,
    prediction_finished - generate_finished,
)

# visualizing model results
if VISUALIZE_RESULTS:
    model.plot_drift_model()
    model.plot_population_model()
plotting_finished = time.time()
eval_times["plotting"] = (
    VISUALIZE_RESULTS,
    plotting_finished - prediction_finished,
)

# sensitivity analysis
if TEST_SENSITIVITY:
    model.drift_sensitivity_analysis()
    model.population_sensitivity_analysis()
sensitivity_finished = time.time()
eval_times["sensitivity"] = (
    TEST_SENSITIVITY,
    sensitivity_finished - plotting_finished,
)

print("Evaluation times:")
for k in eval_times:
    print(f"{k} (shortcutted: {eval_times[k][0]}): {eval_times[k][1]}s")
