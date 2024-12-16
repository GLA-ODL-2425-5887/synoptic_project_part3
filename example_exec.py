from simulation import Simulation
from settings import Settings

settings = Settings(n=200)
s = Simulation(settings)
s.run()
df_summary = s.activity_summary(15)
print(df_summary)
