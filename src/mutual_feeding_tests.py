from coupledgp import generate_data, CoupledGPModel
import time
import os

base_path = os.path.dirname(__file__)

start_time = time.time()
X, Y = generate_data(100)
generate_finished = time.time()
model = CoupledGPModel(X, Y)
model.optimize()
model.save_models(
    "coupledgp/tests/drift_model", "coupledgp/tests/population_model"
)
model_finished = time.time()

print(
    f"Generate Data: {generate_finished - start_time}s, Model Training: {model_finished - generate_finished}s"
)
