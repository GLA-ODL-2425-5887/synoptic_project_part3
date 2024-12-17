from simulation import Simulation
from settings import Settings

settings = Settings(t=500, n=100, g=(2,2,2,3), g_l=None)
s = Simulation(settings)
s.run()
df_summary = s.activity_summary(15)
print(df_summary)
print(s.most_polarised())
print(s.most_polarised_day())
s.plot_network(15)
