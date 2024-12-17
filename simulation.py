from multipartite.base import Multipartite
from opinion_model.individual import NegativeIndividual
from opinion_model.individual import PositiveIndividual
from opinion_model.individual import UnbiasedIndividual
from opinion_model.location import Location
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

class Simulation:
    """
    A class representing an opinion model simulation.
    """
    categories = {
        "ExtremeAgainst": (0.0, 0.2),
        "ModerateAgainst": (0.2, 0.4),
        "Neutral": (0.4, 0.6),
        "ModerateFor": (0.6, 0.8),
        "ExtremeFor": (0.8, 1.0)
    }
    
    @classmethod
    def _categorise_opinion(cls, opinion):
        for category, (low, high) in Simulation.categories.items():
            if low <= opinion < high:
                return category
        return None
    
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
        self.n = settings.n     # Number of individuals in simulation
        self.n_l = settings.n_l # Locations for individuals
        self.g = settings.g     # Number of activity periods and activities
        self.g_l = settings.g_l # Locations for activities
        
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
            
        # Configuration for computing opinions
        self.beta_spread = settings.beta_spread
        self.beta_update = settings.beta_update
        self.gamma_extr = settings.gamma_extr
        
        # Generate list of individual types for simulation using probabilities defined above
        choices = np.random.choice(
            ["NegativeIndividual", "PositiveIndividual", "UnbiasedIndividual"], 
            self.n, True, p=[self.alpha_neg, self.alpha_pos, self.alpha_unbiased]
        )
        
        self.individuals = {}  # Use a dictionary to hold a map of the individuals created by the simulation
        self.activities = {}  # Use a dictionary to hold a map the activities created by the simulation
        
        # Define a Multipartite Graph with len(g)+1 groups
        # Group 0 holds individuals, and Groups 1 to len(g)] represent activity periods
        self.op_mod_graph = Multipartite(len(self.g)+1)
        
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
                
        if self.n_l is not None:
            if len(self.n_l) != self.n:
                raise ValueError("Mismatch between n_l dimensions and number of individuals.")
            
            # Loop through individuals and assign locations sequentially using n_l values
            for i, individual_id in enumerate(self.individuals):
                self.individuals[individual_id].location = Location(*self.n_l[i])    
        
        # Iterate through periods and assign activities with locations
        for period_index, activity_count in enumerate(self.g, start=1):  
            # Create activities for the current period
            activities = [
                self.op_mod_graph.add_node(group=period_index)
                for _ in range(activity_count)
            ]

            # Assign locations based on g_l if provided, otherwise use random locations
            if self.g_l is not None:
                if len(self.g_l) < len(self.g) or len(self.g_l[period_index - 1]) < activity_count:
                    raise ValueError("Mismatch between g_l dimensions and activity periods or counts.")

                # Assign locations from g_l for the current period
                self.activities.update({
                    activity: Location(*self.g_l[period_index - 1][i])
                    for i, activity in enumerate(activities)
                })
            else:
                # Assign random locations if g_l is not provided
                self.activities.update({
                    activity: Location(random.uniform(0, 1), random.uniform(0, 1))
                    for activity in activities
                })

        # Allocate each individual to one activity in each activity period
        for individual_id in self.op_mod_graph.get_nodes(group=0):
            x, y = self.individuals[individual_id].location.x, self.individuals[individual_id].location.y

            for period_index in range(1, len(self.g) + 1):
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
        opinions = []
        for individual_id in self.op_mod_graph.get_nodes(group=0):
            individual = self.individuals[individual_id]
            if time not in individual.opinion_history:
                raise ValueError(f"Time key {time} does not exist in opinion history for individual {individual_id}.")
            opinions.append((individual_id, individual.opinion_history[time]))
    
        return pd.DataFrame(opinions, columns=["id", "opinion"])
    
    def get_multipartite_graph(self):
        return self.op_mod_graph
    
    def _perform_opinion_update(self, opinions):
        """
        Perform an opinion update for individuals in an activity using numpy broadcasting.
        
        Args:
            opinions (pd.DataFrame): A DataFrame with a single column "opinion" containing opinions of individuals.
        
        Returns:
            pd.DataFrame: A DataFrame containing the updated opinions.
        """
        # Convert opinions DataFrame to a numpy array for computation
        opinion_values = opinions["opinion"].values  # Shape: (num_opinions,)
        num_opinions = len(opinion_values)
    
        # Placeholder for new opinions
        new_opinions = opinion_values.copy()
    
        for i, phi_i in enumerate(opinion_values):
            # Determine if the individual becomes completely convinced
            if np.random.rand() < self.gamma_extr:
                if phi_i == 0.5:
                    new_opinions[i] = np.random.randint(2)  # Either 0 or 1
                elif phi_i < 0.5:
                    new_opinions[i] = 0
                else:
                    new_opinions[i] = 1
            else:
                # Proceed with the regular opinion update process
                # Create a matrix of differences: (phi_j - phi_i) for all i, j
                opinion_diff = opinion_values.reshape(1, -1) - opinion_values.reshape(-1, 1)  # Shape: (num_opinions, num_opinions)
    
                # Compute the exponential decay term: exp(-beta_spread * abs(phi_i - phi_j))
                exp_decay = np.exp(-self.beta_spread * np.abs(opinion_diff))  # Shape: (num_opinions, num_opinions)
    
                # Compute the sum for each phi_i, excluding self-contribution (diagonal)
                update_terms = np.sum(opinion_diff * exp_decay, axis=1) / num_opinions  # Shape: (num_opinions,)
    
                # Update opinion using the formula
                new_opinions[i] = phi_i + self.beta_update * update_terms[i]
    
        # Clamp updated opinions to the range [0, 1]
        new_opinions = np.clip(new_opinions, 0, 1)
    
        # Convert the updated opinions back to a DataFrame
        return pd.DataFrame(new_opinions, columns=["updated_opinion"])

    
    def _perform_opinion_activity(self, activity, t):
        """
        Perform opinion update for individuals attending a given activity at time t.
    
        Parameters:
            activity: Node identifier for the activity.
            t: Current time step.
        """
        # Find the individuals attending the activity
        individual_ids = [edge[0] for edge in self.op_mod_graph.get_edges_node(activity)]
    
        # Get opinions of individuals for the previous time step
        opinions = self.get_opinion(t - 1)
    
        # Filter opinions to include only the relevant individuals
        opinions = opinions[opinions["id"].isin(individual_ids)].reset_index(drop=True)
    
        # Check if there are any opinions to process
        if opinions.empty:
            raise ValueError(f"No individuals attended activity {activity} at time {t}.")
    
        # Pass only the "opinion" column to the update method
        updated_opinions = self._perform_opinion_update(opinions[["opinion"]])
    
        # Update each individual's opinion history with the new opinions
        for idx, individual_id in enumerate(individual_ids):
            updated_opinion = updated_opinions.iloc[idx, 0]
            self.individuals[individual_id].opinion = updated_opinion
            self.individuals[individual_id].opinion_history[t] = updated_opinion          
      
    def run(self):
        
        num_activity_periods = len(self.g)  
        
        # For each day in total number of time periods, t...
        for t in range(1, self.t+1):
       
            # Determine the group to process (done in a circular fashion so that
            # only one activity period is processed per time period)
            activity_period_index = (t-1) % num_activity_periods 
            current_activity_period = 1 + activity_period_index  
            
            # Retrieve activities for the specified group
            activities = {
                activity_id: self.activities[activity_id]
                for activity_id in self.op_mod_graph.get_nodes(group=current_activity_period)
                if activity_id in self.activities
            }
           
            # Iterate through all activity identifiers...
            for activity in activities.keys():
                # Update the opinion of each individual attending each activity
                self._perform_opinion_activity(activity, t)
        
    def plot_network(self, time):
        """
        Plot the state of the population and activities at the given time.
    
        Args:
            time (int): The time point at which to plot the network.
        """
        # Define opinion group ranges and their colors
        opinion_colors = {
            "ExtremeAgainst": (0.0, 0.2, "red"),
            "ModerateAgainst": (0.2, 0.4, "orange"),
            "Neutral": (0.4, 0.6, "gray"),
            "ModerateFor": (0.6, 0.8, "blue"),
            "ExtremeFor": (0.8, 1.0, "green")
        }
    
        # Initialize figure
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Plot individuals
        individuals_x = []
        individuals_y = []
        individual_colors = []
        for individual in self.individuals.values():
            if time in individual.opinion_history:
                opinion = individual.opinion_history[time]
                x, y = individual.location.x, individual.location.y
                individuals_x.append(x)
                individuals_y.append(y)
    
                # Assign a color based on opinion
            color_assigned = False
            for category, (lower, upper, color) in opinion_colors.items():
                if lower <= opinion < upper:
                    individual_colors.append(color)
                    color_assigned = True
                    break

            # Handle missing color (shouldn't happen, but safeguard)
            if not color_assigned:
                individual_colors.append("black")  # Default to black if no range matches

        # Ensure no mismatch in lengths
        assert len(individuals_x) == len(individual_colors), (
            f"Mismatch in sizes: {len(individuals_x)} coordinates but {len(individual_colors)} colors"
            )
        ax.scatter(individuals_x, individuals_y, c=individual_colors, s=50, label="Individuals")
    
        # Plot activities
        activity_x = []
        activity_y = []
        activity_edges = []
        for activity_id, activity in self.activities.items():
            x, y = x, y = activity.x, activity.y
            activity_x.append(x)
            activity_y.append(y)

            # Find participants of the activity
            participants = [edge[0] for edge in self.op_mod_graph.get_edges_node(activity_id)]
            for participant_id in participants:
                participant = self.individuals[participant_id]
                if time in participant.opinion_history:
                    activity_edges.append(((participant.location.x, participant.location.y), (x, y)))
    
        ax.scatter(activity_x, activity_y, c="black", s=100, marker="^", label="Activities")
    
        # Add edges (lines) between participants and activities
        if activity_edges:
            line_segments = LineCollection(activity_edges, colors="black", linewidths=0.5)
            ax.add_collection(line_segments)
    
        # Add legend
        legend_elements = [
            Patch(facecolor=color, edgecolor="black", label=label)
            for label, (_, _, color) in opinion_colors.items()
        ]
        legend_elements.append(Patch(facecolor="black", edgecolor="black", label="Activities"))
        ax.legend(handles=legend_elements, loc="upper right")
    
        # Set plot limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("X Location")
        ax.set_ylabel("Y Location")
        ax.set_title(f"Network State at Time {time}")
    
        plt.show()

    
    def chart(self):
        pass

    def most_polarised(self):
        """
        Calculate the most polarised moments in the simulation, returning:
        - The maximal number of individuals who were ExtremeAgainst at any one time.
        - The maximal number of individuals who were ExtremeFor at any one time.
        - The maximal number of individuals in either extreme group at any one time.
    
        Returns:
            dict: A dictionary with three keys:
                  'extreme_against': maximal count of ExtremeAgainst individuals,
                  'extreme_for': maximal count of ExtremeFor individuals,
                  'total_extremes': maximal count of all individuals in extreme groups.
        """
        # Retrieve thresholds from categories
        extreme_against_range = self.categories["ExtremeAgainst"]
        extreme_for_range = self.categories["ExtremeFor"]
    
        max_extreme_against = 0
        max_extreme_for = 0
        max_total_extremes = 0
    
        # Identify all time steps across all individuals
        time_periods = set()
        for individual in self.individuals.values():
            time_periods.update(individual.opinion_history.keys())
        time_periods = sorted(time_periods)  # Ensure sorted order
    
        for t in time_periods:
            num_extreme_against = 0
            num_extreme_for = 0
    
            # Count individuals in extreme categories at time t
            for individual in self.individuals.values():
                if t in individual.opinion_history:
                    opinion = individual.opinion_history[t]
                    if extreme_against_range[0] <= opinion < extreme_against_range[1]:
                        num_extreme_against += 1
                    elif extreme_for_range[0] <= opinion <= extreme_for_range[1]:
                        num_extreme_for += 1
    
            total_extremes = num_extreme_against + num_extreme_for
    
            # Update maximums
            max_extreme_against = max(max_extreme_against, num_extreme_against)
            max_extreme_for = max(max_extreme_for, num_extreme_for)
            max_total_extremes = max(max_total_extremes, total_extremes)
    
        return {
            "extreme_against": max_extreme_against,
            "extreme_for": max_extreme_for,
            "total_extremes": max_total_extremes,
        }

    def most_polarised_day(self):
        """
        Find the day with the maximum polarisation based on:
        - The maximal number of individuals who were ExtremeAgainst,
        - The maximal number of individuals who were ExtremeFor,
        - The maximal number of individuals in either extreme group.
    
        Returns:
            dict: A dictionary with keys:
                  'extreme_against_day': day with max ExtremeAgainst count,
                  'extreme_for_day': day with max ExtremeFor count,
                  'total_extremes_day': day with max total extremes count.
        """
        # Retrieve thresholds from categories
        extreme_against_range = self.categories["ExtremeAgainst"]
        extreme_for_range = self.categories["ExtremeFor"]
    
        max_extreme_against = 0
        max_extreme_for = 0
        max_total_extremes = 0
        extreme_against_day = None
        extreme_for_day = None
        total_extremes_day = None
    
        # Identify all time steps across all individuals
        time_periods = set()
        for individual in self.individuals.values():
            time_periods.update(individual.opinion_history.keys())
        time_periods = sorted(time_periods)  # Ensure sorted order
    
        for t in time_periods:
            num_extreme_against = 0
            num_extreme_for = 0
    
            # Count individuals in extreme categories at time t
            for individual in self.individuals.values():
                if t in individual.opinion_history:
                    opinion = individual.opinion_history[t]
                    if extreme_against_range[0] <= opinion < extreme_against_range[1]:
                        num_extreme_against += 1
                    elif extreme_for_range[0] <= opinion <= extreme_for_range[1]:
                        num_extreme_for += 1
    
            total_extremes = num_extreme_against + num_extreme_for
    
            # Update maximums and their corresponding days
            if num_extreme_against > max_extreme_against:
                max_extreme_against = num_extreme_against
                extreme_against_day = t
    
            if num_extreme_for > max_extreme_for:
                max_extreme_for = num_extreme_for
                extreme_for_day = t
    
            if total_extremes > max_total_extremes:
                max_total_extremes = total_extremes
                total_extremes_day = t
    
        return {
            "extreme_against_day": extreme_against_day,
            "extreme_for_day": extreme_for_day,
            "total_extremes_day": total_extremes_day,
        }

    
    def activity_summary(self, t):
        """
        Summarizes opinions of individuals at time `t` into specified categories, grouped by activity.
        
        Parameters:
            t (int): The time step for which opinions are being summarized.
            
        Returns:
            pd.DataFrame: A summary DataFrame grouped by activity, with counts for each category.
        """
        # Initialize an empty list to collect data
        data = []
        
        # For each activity...
        for activity in self.activities:
            # Find all activity participants
            individual_ids = [edge[0] for edge in self.op_mod_graph.get_edges_node(activity)]
               
            # Iterate through individuals in the activity and collect their opinions at time `t`
            for individual_id in individual_ids:
                if t in self.individuals[individual_id].opinion_history:
                    data.append({
                        "Activity": activity,
                        "id": individual_id,
                        "opinion": self.individuals[individual_id].opinion_history[t]
                    })
    
        # Create a DataFrame from the collected data
        df_summary = pd.DataFrame(data)
        
        # If no data was collected, return an empty DataFrame
        if df_summary.empty:
            return pd.DataFrame(columns=["Activity"] + list(Simulation.categories.keys()))
    
        # Categorize opinions
        df_summary["category"] = df_summary["opinion"].apply(Simulation._categorise_opinion)
        
        # Group by Activity and category, and calculate counts
        df_grouped = (
            df_summary.groupby(["Activity", "category"])
            .size()
            .reset_index(name="count")  # Reset the index to flatten the grouped result
        )
        
        # Pivot the DataFrame to wide format
        df_pivot = df_grouped.pivot_table(
            index = "Activity",  # Rows are activities
            columns = "category",  # Columns are categories
            values = "count",  # Values are counts
            fill_value = 0  # Replace missing values with 0
        )
        
        # Ensure categories appear in the desired order, setting missing categories to 0
        all_categories = list(Simulation.categories.keys()) 
        df_pivot = df_pivot.reindex(columns=all_categories, fill_value=0)
        
        # Convert counts to integers to remove any decimal points
        df_pivot = df_pivot.astype(int)
        return df_pivot
        
    def individual_summary(self):
        pass
    
    def friendship_summary(self):
        pass
    
    def friendship_similarlity_chart(self):
        pass
