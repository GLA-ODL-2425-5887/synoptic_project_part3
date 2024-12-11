class Node:
    """
     A class representing a node within a multipartite graph.
     Each node has a numeric identifer and an x,y location (0 <= x,y <= 1)
    """
    _id = 0
    
    def __init__(self, x=0, y=0):
        self.id = Node._id
        
        Node._id += 1
        
    def __repr__(self):
        return f"Node {self.id} ({self.location.x},{self.location.y})"