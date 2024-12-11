from multipartite.node import Node

class Multipartite:
    """
    A class representing a multipartite graph with no weights and directions.
    Now uses dictionaries for more efficient lookups.
    """
    
    def __init__(self, number_of_groups):
        if not isinstance(number_of_groups, int):
            raise ValueError("The number of groups must be an integer.")
        if number_of_groups <= 0:
            raise ValueError("The number of groups must be greater than 0.")
        
        self.number_of_groups = number_of_groups
        self.nodes = {}  # use node.id as a hash to a group number
        self.edges = []  # Use a list to store edges

    def __str__(self):
        return f"Multipartite with {self.number_of_groups} groups"

    def add_node(self, group):
        if 0 <= group < self.number_of_groups:
            node = Node()
            self.nodes[node.id] = group  # Store node with group information
            return node.id # return the hash
        else:
            raise ValueError(f"Group {group} does not exist.")
    
    def node_exists(self, node_identifier):
        return node_identifier in self.nodes  
    
    def get_nodes(self, group=None):
        if group is not None:
            if 0 <= group < self.number_of_groups:
                return [node_id for node_id, grp in self.nodes.items() if grp == group]
            else:
                raise ValueError(f"Group {group} does not exist.")
        else:
            return list(self.nodes.keys())
    
    def num_nodes(self, group=None):
        if group is not None:
            if 0 <= group < self.number_of_groups:
                return sum(1 for grp in self.nodes.values() if grp == group)
            else:
                raise ValueError(f"Group {group} does not exist.")
        else:
            return len(self.nodes)
    
    def get_group(self, node_identifier):
        if node_identifier in self.nodes:
            return self.nodes[node_identifier]
        else:
            raise ValueError(f"Node {node_identifier} does not exist.")
    
    def add_multipartite_edge(self, node_identifier1, node_identifier2):
        if not self.node_exists(node_identifier1) or not self.node_exists(node_identifier2):
            raise ValueError("One or both nodes do not exist.")
        
        group1 = self.get_group(node_identifier1)
        group2 = self.get_group(node_identifier2)
        
        if group1 != group2:
            edge = (min(node_identifier1, node_identifier2), max(node_identifier1, node_identifier2))
            if edge not in self.edges:  # Avoid duplicate edges
                self.edges.append(edge)  
            else:
             raise ValueError(f"Nodes {node_identifier1} and {node_identifier2} already exists as an edge.") 
        else:
            raise ValueError(f"Nodes {node_identifier1} and {node_identifier2} are in the same group.")
    
    def remove_multipartite_edge(self, node_identifier1, node_identifier2):
        edge = (min(node_identifier1, node_identifier2), max(node_identifier1, node_identifier2))
        if edge in self.edges:
            self.edges.remove(edge)
        else:
            raise ValueError(f"Edge between Node {node_identifier1} and Node {node_identifier2} does not exist.")

    def get_edges(self):
        return list(self.edges)

    def get_edges_node(self, node_identifier):
        return [edge for edge in self.edges if node_identifier in edge]

    def get_edges_nodes(self, collection_of_node_identifiers, exclusive=False):
        edges = []
        node_set = set(collection_of_node_identifiers)
        
        for edge in self.edges:
            node1, node2 = edge
            if exclusive:
                if node1 in node_set and node2 in node_set:
                    edges.append(edge)
            else:
                if node1 in node_set or node2 in node_set:
                    edges.append(edge)
        
        return edges
    
    def get_edges_group(self, group1=None, group2=None):
        edges = []
        for edge in self.edges:
            node1, node2 = edge
            group_node1 = self.get_group(node1)
            group_node2 = self.get_group(node2)
            
            if group1 is not None and group2 is not None:
                if (group_node1 == group1 and group_node2 == group2) or (group_node1 == group2 and group_node2 == group1):
                    edges.append(edge)
            elif group1 is not None:
                if group_node1 == group1 or group_node2 == group1:
                    edges.append(edge)
            else:
                edges.append(edge) 
                
        return edges
    
    def has_edge(self, node_identifier1, node_identifier2):
        edge = (min(node_identifier1, node_identifier2), max(node_identifier1, node_identifier2))
        return edge in self.edges
    
    def remove_node(self, node_identifier):
        if not self.node_exists(node_identifier):
            raise ValueError(f"Node {node_identifier} does not exist.")
        
        edges_to_remove = [edge for edge in self.edges if node_identifier in edge]
        for edge in edges_to_remove:
            self.edges.remove(edge)
        
        del self.nodes[node_identifier]
    
    def num_edges(self):
        return len(self.edges)

    def density(self):
        num_edges = len(self.edges)
        total_possible_edges = self._get_total_possible_edges()
        if total_possible_edges == 0:
            return 0
        return num_edges / total_possible_edges
    
    def _get_total_possible_edges(self):
        total_possible_edges = 0
        for c1 in range(self.number_of_groups):
            c1_nodes = len(self.get_nodes(c1))
            for c2 in range(c1 + 1, self.number_of_groups):
                c2_nodes = len(self.get_nodes(c2))
                total_possible_edges += c1_nodes * c2_nodes
        return total_possible_edges