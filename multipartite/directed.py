from multipartite.base import Multipartite

class DirectedMultipartite(Multipartite): 
    def __str__(self):
        return f"Directed multipartite with {self.number_of_groups} groups"
    
    def add_multipartite_edge(self, node_identifier1, node_identifier2):
        if not self.node_exists(node_identifier1) or not self.node_exists(node_identifier2):
            raise ValueError("One or both nodes do not exist.")
        
        group1 = self.get_group(node_identifier1)
        group2 = self.get_group(node_identifier2)
        
        if group1 != group2:
            edge = (node_identifier1, node_identifier2)
            if edge not in self.edges:
                self.edges.append(edge)  
            else:
             raise ValueError(f"Nodes {node_identifier1} and {node_identifier2} already exists as an edge.")   
        else:
            raise ValueError(f"Nodes {node_identifier1} and {node_identifier2} are in the same group.")
    
    def remove_multipartite_edge(self, node_identifier1, node_identifier2):
        edge = (node_identifier1, node_identifier2)
        if edge in self.edges:
            self.edges.remove(edge)
        else:
            raise ValueError(f"Edge between Node {node_identifier1} and Node {node_identifier2} does not exist.")

    def get_edges_node(self, node_identifier, direction=None):
        if direction == "from":
            return [edge for edge in self.edges if edge[0] == node_identifier]
        elif direction == "to":
            return [edge for edge in self.edges if edge[1] == node_identifier]
        else:
            return [edge for edge in self.edges if node_identifier in edge]

    def get_edges_nodes(self, collection_of_node_identifiers, exclusive=False, direction=None):
        edges = []
        node_set = set(collection_of_node_identifiers)
        
        for edge in self.edges:
            node1, node2 = edge
            if exclusive:
                if node1 in node_set and node2 in node_set:
                    edges.append(edge)
            else:
                if direction == "from":
                    if node1 in node_set:
                        edges.append(edge)
                elif direction == "to":
                    if node2 in node_set:
                        edges.append(edge)
                else:
                    if node1 in node_set or node2 in node_set:
                        edges.append(edge)
        return edges
    
    def has_edge(self, node_identifier1, node_identifier2):
        edge = (node_identifier1, node_identifier2)
        return edge in self.edges
    
    def density(self):
        num_edges = len(self.edges)
        total_possible_edges = 2 * super()._get_total_possible_edges()
        if total_possible_edges == 0:
            return 0
        return num_edges / total_possible_edges
    

