from opinion_model.location import Location

class Activity():
    """
    A class representing an activity that is part of an opinion model
    """
    def __init__(self, x=0, y=0):
        self.location = Location(x, y)
        
        # List of time periods that the activity was active in
        self.run_history = [0] 
        
    def __repr__(self):
        return f"{self.location}"
        
    def __str__(self):
        return f"{self.location}" 