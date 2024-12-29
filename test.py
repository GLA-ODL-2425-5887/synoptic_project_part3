from simulation import Simulation
from settings import Settings
import matplotlib.pyplot as plt

def plot_opinion_distribution(opinions, step):
    plt.hist(opinions, bins=10, range=(0, 1), alpha=0.7)
    plt.title(f"Opinion Distribution at Step {step}")
    plt.xlabel("Opinion")
    plt.ylabel("Count")
    plt.show()


settings = Settings(beta_update=0)
simulation = Simulation(settings)
opinions = simulation.get_opinion(0)["opinion"]
plot_opinion_distribution(opinions, 0)
print(simulation.activity_summary(0))
simulation.run()
opinions = simulation.get_opinion(500)["opinion"]
plot_opinion_distribution(opinions, 500)
simulation.chart()
print(simulation.activity_summary(500))

