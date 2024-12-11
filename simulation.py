from multipartite.base import Multipartite
from opinion_model.individual import NegativeIndividual
from opinion_model.individual import PositiveIndividual
from opinion_model.individual import UnbiasedIndividual
from opinion_model.location import Location
from collections import Counter
import numpy as np
import random

class Simulation:
    """
    A class representing an opinion model simulation.
    """
    
    @classmethod
    def ensemble_statistics(cls):
        pass
    
    def __init__(self, settings):
        """
        Initialise the simulation as follows:
            1. Load settings
            2. Generate a list of Individuals with a range of randomly assigned (based on settings) opinions
            3. Add these Indivduals to a multipartite graph
            4. Add Activity Period and associated Activities nodes to the graph
            5. Allocate Individuals to Activities, storing these relationships as edges in the graph
        
        Parameters:
            settings (Settings)
        """
        
        self.n = settings.n  # Number of individuals in simulation
        self.g = settings.g  # Number of activity periods and activities
        
        if(self.g[0] <= 0 or self.g[1] <= 0):
            raise ValueError("Number of activity periods and activies must be greater than zero")
              
        # Number of days to run the simulation
        self.t = settings.t
        
        # Probabilities for each type of individual
        self.alpha_pos = settings.alpha_pos
        self.alpha_neg = settings.alpha_neg
        self.alpha_unbiased = 1 - self.alpha_neg - self.alpha_pos 
        
        # Configuration values used for computing activity selection
        self.lambda_dist = settings.lambda_dist
        if self.lambda_dist <= 0:
            raise ValueError("Lambda distance must be greater than zero")
        
        # Generate list of individual types for simulation using probabilities defined above
        choices = np.random.choice(
            ["NegativeIndividual", "PositiveIndividual", "UnbiasedIndividual"], 
            self.n, True, p=[self.alpha_neg, self.alpha_pos, self.alpha_unbiased]
        )
        
        self.individuals = {}  # Use a dictionary to hold the individuals created by the simulation
        self.activities = {}  # Use a dictionary to hold the activities created by the simulation
        
        # Define a Multipartite Graph with g[0]+1 groups
        # Group 0 holds individuals, and Groups 1 to g[0] represent activity periods
        self.op_mod_graph = Multipartite(self.g[0]+1)
        
        # Store the individuals in the first group of the opinion model
        for i in choices:
            id = self.op_mod_graph.add_node(group=0)
        
            # Create the corresponding type of individual and store them indexed by node ID
            if i == "NegativeIndividual":
                self.individuals[id] = NegativeIndividual(settings)
            elif i == "PositiveIndividual":
                self.individuals[id] = PositiveIndividual(settings)
            else:
                self.individuals[id] = UnbiasedIndividual(settings)
        
        # Iterate through activity periods and add their activities
        for period_index in range(1, self.g[0]+1):  
        
            # Add activities for the current period and store their locations
            activities = [
                self.op_mod_graph.add_node(group=period_index)
                for _ in range(self.g[1])
            ]
            self.activities.update({
                activity: Location(random.uniform(0, 1), random.uniform(0, 1))
                for activity in activities
            })
            
        # Allocate each individual to one activity in each activity period
        for individual_id in self.op_mod_graph.get_nodes(group=0):
            x, y = self.individuals[individual_id].location.x, self.individuals[individual_id].location.y

            for period_index in range(1, self.g[0] + 1):
                # Compute probabilities for the individual's selection of activities in the current period
                activity_probabilities = self._compute_activity_probabilities(x, y, period_index)

                # Extract activities and their probabilities
                activities, probs = zip(*activity_probabilities)

                # Randomly sample an activity based on relative probabilities
                selected_activity_id = np.random.choice(activities, p=probs)

                # Add an edge between the individual and the selected activity
                self.op_mod_graph.add_multipartite_edge(individual_id, selected_activity_id)
   
    def _compute_activity_probabilities(self, x, y, group):
        """
        Calculate probabilities for assigning a node to activities based on distances, 
        then return a list of activity node identifiers and probabilities.
        
        Parameters:
            x (float): x-coordinate of the individual.
            y (float): y-coordinate of the individual.
            group (int): The activity group index.
            
        Returns:
            List[Tuple[int, float]]: List of (activity_id, probability) tuples, sorted by probability.
        """
        # Retrieve activities for the specified group
        activities = {
            activity_id: self.activities[activity_id]
            for activity_id in self.op_mod_graph.get_nodes(group=group)
            if activity_id in self.activities
        }
        
        # Set location of the individual of interest
        individual_location = Location(x, y)
        
        # Calculate Euclidean distances from the individual's location to each activity's location
        distances = np.array([Location.distance(individual_location, activity)
                              for activity in activities.values()])
        
        # Calculate exponential terms
        exp_terms = np.exp(-self.lambda_dist * distances)
        
        # Calculate normalised probabilities
        probabilities = exp_terms / np.sum(exp_terms)
        
        # Pair activity IDs with probabilities
        return list(zip(activities.keys(), probabilities))  

    def get_opinion(self, time):
        pass
    
    def get_multipartite_graph(self):
        return self.op_mod_graph
    
    def _perform_opinion_update(self, opinions):
        pass
    
    def _perform_opinion_activity(self, activity, t):
        pass
    
    def run(self):
        
        # For each day in total number of days, t...
        for day in range(1, self.t):
            
            # Randomise the order of the individuals
            np.random.shuffle(self.individuals)
            
            # then set the opinion of each individual for that day
            for individual in self.individuals:
                individual.set_opinion()
        
    def plot_network(self):
        pass
    
    def chart(self):
        pass

    def _most_polarised(self):
        
        # Initialize empty counters for each day
        extreme_counts = {
            "ExtremeAgainst": [0] * self.t,
            "ExtremeFor": [0] * self.t,
            "ExtremeEither": [0] * self.t
        }

        # Aggregate daily counts across all individuals
        for day, opinions in enumerate(zip(*(individual.opinion_history for individual in self.individuals))):
            counts = Counter(opinions)
            extreme_counts["ExtremeFor"][day] = counts[1]
            extreme_counts["ExtremeAgainst"][day] = counts[0]
            extreme_counts["ExtremeEither"][day] = counts[0] + counts[1]
    
        return extreme_counts

    def most_polarised(self):
        extreme_counts = self._most_polarised()
        
        max_extreme_count = {
            "ExtremeAgainst": max(extreme_counts["ExtremeAgainst"]),
            "ExtremeFor": max(extreme_counts["ExtremeFor"]),
            "ExtremeEither": max(extreme_counts["ExtremeEither"])
        }
    
        return max_extreme_count
    
    def most_polarised_day(self):
        extreme_counts = self._most_polarised()
    
        max_extreme_day = {
            "ExtremeAgainst": extreme_counts["ExtremeAgainst"].index(max(extreme_counts["ExtremeAgainst"])),
            "ExtremeFor": extreme_counts["ExtremeFor"].index(max(extreme_counts["ExtremeFor"])),
            "ExtremeEither": extreme_counts["ExtremeEither"].index(max(extreme_counts["ExtremeEither"]))
        }
    
        return max_extreme_day

    
    def individual_summary(self):
        pass
    
    def friendship_summary(self):
        pass
    
    def friendship_similarlity_chart(self):
        pass
