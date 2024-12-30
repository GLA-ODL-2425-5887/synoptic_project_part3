import math

class Location:
    """
    A class representing a location using coordinates x and y.
    """
    @classmethod
    def distance(cls, a, b):
        """
        Return the Euclidean distance between two locations, a and b.
        """
        return math.sqrt(((a.x - b.x)**2) + ((a.y - b.y)**2))
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x},{self.y})"

