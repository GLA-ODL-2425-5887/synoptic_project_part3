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
        self.opinion = round(np.random.uniform(lower, upper), 3) #Keep everything to 3 decimal places
        
    def __repr__(self):
        return f"{self.id}"
        
    def __str__(self):
        return f"{self.id} ({self.location.x},{self.location.y}) with opinion {self.opinion}"  
        
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
    
    
 # def set_opinion(self):
 #     new_opinion = self.opinion # Opinion might not change
     
 #     # See if they can become completely convinced...
 #     if np.random.rand() < self.gamma_extr:
 #         if self.opinion == 0.5:
 #             new_opinion = np.random.randint(2) # either 0 or 1
 #         elif self.opinion < 0.5:
 #             new_opinion = 0
 #         else:
 #             new_opinion = 1
 #     else: # they can't, so check each friend's opinion...
         
 #         # ... but only if they have any friends! (Avoid divide by zero)
 #         if len(self.connections) > 0:
         
 #             # Determine the average of the opinions of the individual's connections
 #             average_opinion = sum([ (connection.opinion - self.opinion) * (math.e**(-self.beta_spread))
 #                            for connection in self.connections ]) / len(self.connections)
             
 #             # Use this to generate a new opinion
 #             new_opinion = self.opinion + (self.beta_update * average_opinion)

 #     # Update opinion history (we want a value per day for each individual)
 #     new_opinion = max(0, min(1, round(new_opinion, 3))) # Keep opinion between 0 and 1
 #     self.opinion_history.append(new_opinion)
 #     self.opinion = new_opinion
     
 # def add_connection(self, individual):
 #     self.connections.append(individual) 