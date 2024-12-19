from simulation import Simulation
from settings import Settings

settings = Settings(n=20)
simulation = Simulation(settings)
simulation.run()


# Use default predictors
model_full = simulation.fit_regression_model()

# Use only specific predictors from list...
# ["InitialOpinion", "DistanceToCentre", "NearestActivityDistance"]
selected_predictors = ["InitialOpinion", "DistanceToCentre"]
model_reduced = simulation.fit_regression_model(predictors=selected_predictors)

# Summary of the fitted model
print(model_full.summary())
