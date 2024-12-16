class Settings:
    """
    A class used to store configuration parameters for an opinion model simulation
    """
    
    def __init__(self, 
                 t = 500,               # Duration (number of simulated days) for which the simulation is run
                 n = 200,               # Number of individuals in the simulation
                 
                 # List of (x,y) locations for the individuals in the simulation
                 n_l = None,              # Defaults to empty list, so individuals will be assigned random locations
                 
                 alpha_pos = 0.25,      # Probability that an individual will be of the positive type
                 alpha_neg = 0.25,      # Probability that an individual will be of the negative type
                 lambda_dist = 1,       # Parameter which controls how far people will travel for activities
                 beta_update = 0.01,    # Controls how much an individuals opinion moves towards their contacts
                 beta_spread = 0.01,    # Controls how much an individuals listens to those with a different opinion to theirs
                 gamma_extr = 0.005,    # Probability that unbiased/positive/negative individuals become completely convinced
                 
                 # tuple of integers with:
                 #     the length of Gn representing the number of activity periods
                 #     the ith element representing the number of activities/classes in the ith period
                 g = (4,4),
                 
                 # List of (x,y) locations for the activities in the simulation
                 g_l = [
                         [(0,0),(0,1),(1,0),(1,1)],
                         [(1,0.5),(0.5,0),(1,0.25),(0.45,1)]
                ]
    ):
        
        self.t = t
        self.n = n
        self.n_l = n_l
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.lambda_dist = lambda_dist
        self.beta_update = beta_update
        self.beta_spread = beta_spread
        self.gamma_extr = gamma_extr
        self.g_l = g_l
        self.g = g
        