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
USE_DATA_PATH = True
USE_MODEL_PATH = True

# timing initialization
start_time = time.time()
eval_times = dict()

# generate training data
if USE_DATA_PATH:
    X, Y = load_data_from_file(
        "./src/coupledgp/tests/x-100.npy", "./src/coupledgp/tests/y-100.npy"
    )
else:
    X, Y = generate_data(100, "./src/coupledgp/tests/")
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
design = LatinDesign(coupled_space)
test_inputs = design.get_samples(5)
test_outputs = model.predict(test_inputs)
print(test_outputs)
prediction_finished = time.time()
eval_times["prediction"] = (False, prediction_finished - generate_finished)

# visualizing genetic drift model results

# visualizing population model results

# visualizing coupled model results

# sensitivity analysis for genetic drift model

# sensitivity analysis for population model

# sensitivity analysis for coupled model

print("Evaluation times:")
for k in eval_times:
    print(f"{k} (use_path: {eval_times[k][0]}): {eval_times[k][1]}s")
