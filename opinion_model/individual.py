from opinion_model.location import Location
import numpy as np

class Individual():
    """
    A class representing an individual that is part of an opinion model
    """
    def __init__(self, settings, lower=0, upper=1, x=0, y=0):
        
        # Settings
        self.beta_update = settings.beta_update
        if self.beta_update < 0:
            raise ValueError("Beta update must not be less than zero")
        self.beta_spread = settings.beta_spread
        if self.beta_spread <= 0:
            raise ValueError("Beta update must be greater than zero")
        self.gamma_extr = settings.gamma_extr
        
        # State
        self.location = Location(x, y)
        self.opinion = np.random.uniform(lower, upper)
        self.opinion_history = {0 : self.opinion} # Store initial opinion in opinion history dictionary indexed by time period
        
    def __repr__(self):
        return f"{self.id}"
        
    def __str__(self):
        return f"{self.id} {self.location} with opinion {self.opinion}"  
        
class PositiveIndividual(Individual):
    """
    A class representing an individual with a positive opinion
    """
    lower = 0.75
    upper = 1
    
    def __init__(self, settings):
        super().__init__(settings, PositiveIndividual.lower, PositiveIndividual.upper)
    
    def __str__(self):
        return f"PositiveIndividual: {super().__str__()}"
        
class NegativeIndividual(Individual):
    """
    A class representing an individual with a negative opinion
    """
    lower = 0
    upper = 0.25
    
    def __init__(self, settings):
        super().__init__(settings, NegativeIndividual.lower, NegativeIndividual.upper)
    
    def __str__(self):
        return f"NegativeIndividual: {super().__str__()}"
    
class UnbiasedIndividual(Individual):
    """
    A class representing an unbiased individual that is 
    """
    lower = 0
    upper = 1
    
    def __init__(self, settings):
        super().__init__(settings, UnbiasedIndividual.lower, UnbiasedIndividual.upper)
    
    def __str__(self):
        return f"UnbiasedIndividual: {super().__str__()}"