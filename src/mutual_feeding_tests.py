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
USE_MODEL_PATH = True

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
        "./src/coupledgp/tests/x-sim-20-100.npy",
        "./src/coupledgp/tests/y-sim-20-100.npy",
    )
else:
    X, Y = generate_data(
        20, 100, "./src/coupledgp/tests/"
    )  # runs simulator for 100 days per sample (total 1000)
generate_finished = time.time()
eval_times["generate"] = (USE_DATA_PATH, generate_finished - start_time)

# training coupled model
model = CoupledGPModel(X, Y)
if USE_MODEL_PATH:
    model.load_models(
        "./src/coupledgp/tests/drift_model_test.npy",
        "./src/coupledgp/tests/population_model_test.npy",
    )
else:
    model.optimize()
    model.save_models(
        "./src/coupledgp/tests/drift_model_test",
        "./src/coupledgp/tests/population_model_test",
    )
model_finished = time.time()
eval_times["training"] = (USE_MODEL_PATH, model_finished - generate_finished)

print("Evaluation times:")
for k in eval_times:
    print(
        f"{k} (shortcutted: {eval_times[k][0]}, graphs: {SHOW_GRAPHS}): {eval_times[k][1]}s"
    )

print(model.drift_emukit.gpy_model)
print(model.pop_emukit.model)

model.plot_training(None, "drift")
model.plot_training(None, "population")
model.plot_training(None, "coupled")
