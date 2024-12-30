from multipartite.base import Multipartite
from opinion_model.individual import NegativeIndividual
from opinion_model.individual import PositiveIndividual
from opinion_model.individual import UnbiasedIndividual
from opinion_model.location import Location
from opinion_model.activity import Activity
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Simulation:
    """
    A class representing an opinion model simulation.
    """
    categories = {
        "ExtremeAgainst": (0.0, 0.2, "red"),
        "ModerateAgainst": (0.2, 0.4, "orange"),
        "Neutral": (0.4, 0.6, "gray"),
        "ModerateFor": (0.6, 0.8, "blue"),
        "ExtremeFor": (0.8, 1.0, "green")
    }
    
    @classmethod
    def _categorise_opinion(cls, opinion):
        for category, (low, high, colour) in Simulation.categories.items():
            if low <= opinion <= high:
                return category
        return None
    
    @classmethod
    def ensemble_statistics(cls, n, settings):
        """
        Perform the simulation n times and generate ensemble statistics with a pair-plot.
        
        Args:
            N (int): Number of simulations to run.
            settings: Settings for each run of the simulation.
            
        Returns:
            pd.DataFrame: DataFrame with ensemble statistics for each simulation.
        """
        # List to store the statistics for each simulation
        ensemble_data = []

        for i in range(n):
            # Create a new simulation instance with the given settings
            simulation = Simulation(settings)

            # Run the simulation
            simulation.run()

            # Collect statistics for this simulation
            stats = {
                "Negative Individuals": simulation._get_negative_count(),
                "Positive Individuals": simulation._get_positive_count(),
                "Max ExtremeAgainst": simulation._get_max_extreme_against(),
                "Max ExtremeFor": simulation._get_max_extreme_for(),
                "Max Involved in any activity": simulation._get_max_involved_activity()
            }

            # Add this simulation's statistics to the ensemble data
            ensemble_data.append(stats)

        # Convert the ensemble data into a pandas DataFrame
        df_ensemble = pd.DataFrame(ensemble_data)

        # Create a pair-plot to visualize the ensemble statistics
        sns.pairplot(df_ensemble)
        plt.suptitle("Ensemble Statistics Pair-Plot", y=1.02)
        plt.show()

        return df_ensemble

    # Helper methods for ensemble statistics
    def _get_negative_count(self):
        """Number of individuals in negative states (ExtremeAgainst or ModerateAgainst)."""
        return sum(
            1 for individual in self.individuals.values()
            if Simulation._categorise_opinion(individual.opinion) in ["ExtremeAgainst", "ModerateAgainst"]
        )

    def _get_positive_count(self):
        """Number of individuals in positive states (ModerateFor or ExtremeFor)."""
        return sum(
            1 for individual in self.individuals.values()
            if Simulation._categorise_opinion(individual.opinion) in ["ModerateFor", "ExtremeFor"]
        )

    def _get_max_extreme_against(self):
        """Maximum number of individuals in ExtremeAgainst at any one time."""
        return max(
            sum(
                Simulation._categorise_opinion(opinion) == "ExtremeAgainst"
                for opinion in [individual.opinion_history[t] for individual in self.individuals.values()]
            )
            for t in range(self.t)
        )

    def _get_max_extreme_for(self):
        """Maximum number of individuals in ExtremeFor at any one time."""
        return max(
            sum(
                Simulation._categorise_opinion(opinion) == "ExtremeFor"
                for opinion in [individual.opinion_history[t] for individual in self.individuals.values()]
            )
            for t in range(self.t)
        )

    def _get_max_involved_activity(self):
        """Maximum number of individuals assigned to any activity."""
        return max(
            len(self.op_mod_graph.get_edges_node(activity))
            for activity in self.activities.keys()
        )    
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
        
        # Used for time summary plot
        self.overall_summary = pd.DataFrame(columns=["Activity", "time"] + list(Simulation.categories.keys()))
        
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
            
            # Assign individual locations sequentially using n_l values, overwriting
            # randomly assigned locations
            for i, individual_id in enumerate(self.individuals):
                self.individuals[individual_id].location = Location(*self.n_l[i])    
        
        # Iterate through activity periods and assign activities with locations
        for period_index, activity_count in enumerate(self.g, start=1):  
            
            # Create activities for the current period
            activities = [
                self.op_mod_graph.add_node(group=period_index)
                for _ in range(activity_count)
            ]
        
            # Assign Activity locations based on g_l if provided, otherwise use random locations
            if self.g_l is not None:
                if len(self.g_l) != len(self.g) or len(self.g_l[period_index - 1]) != activity_count:
                    raise ValueError("Mismatch between g_l dimensions and activity periods or counts.")
        
                # Assign Activity objects with locations from g_l for the current period
                self.activities.update({
                    activity: Activity(*self.g_l[period_index - 1][i])
                    for i, activity in enumerate(activities)
                })
            else:
                # Assign Activity objects with random locations if g_l is not provided
                self.activities.update({
                    activity: Activity(random.uniform(0, 1), random.uniform(0, 1))
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
        
        # Calculate distances (assumption that this is the Euclidean distance)
        # from the individual's location to each activity's location
        distances = np.array([Location.distance(individual_location, activity.location)
                              for activity in activities.values()])
        
        # Calculate exponential terms
        exp_terms = np.exp(-self.lambda_dist * distances)
        
        # Calculate normalised probabilities
        probabilities = exp_terms / np.sum(exp_terms)
        
        # Pair activity IDs with probabilities
        return list(zip(activities.keys(), probabilities))  

    def get_opinion(self, time):
        """
        Retrieve the opinions of all individuals at time t.
        
        Args:
            time (int): Time period.
            
        Returns:
            pd.DataFrame: DataFrame with id and opinion.
        """
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
        
        # Convert opinions DataFrame to a numpy array
        opinion_values = opinions["opinion"].values
        num_opinions = len(opinion_values)
    
        # Work on a copy of the opinions for safety
        new_opinions = opinion_values.copy()
    
        # Random values to decide if individuals become "completely convinced"
        random_values = np.random.rand(num_opinions)
    
        # Identify individuals who are "completely convinced"
        are_convinced = random_values < self.gamma_extr
    
        # If phi_i == 0.5 -> Randomly choose 0 or 1
        new_opinions[are_convinced & (opinion_values == 0.5)] = np.random.randint(
            2, size=np.sum(are_convinced & (opinion_values == 0.5)))
    
        # If phi_i < 0.5 -> Set to 0; If phi_i > 0.5 -> Set to 1
        new_opinions[are_convinced & (opinion_values < 0.5)] = 0
        new_opinions[are_convinced & (opinion_values > 0.5)] = 1
    
        # For the remaining individuals, perform the regular opinion update process
        not_convinced = ~are_convinced
    
        if np.any(not_convinced):  # Proceed only if there are non-convinced individuals
            
            # Create a matrix of differences: (phi_j - phi_i) for all i, j
            opinion_diff = opinion_values.reshape(1, -1) - opinion_values.reshape(-1, 1)
    
            # Compute the exponential decay term
            exp_decay = np.exp(-self.beta_spread * np.abs(opinion_diff))
    
            # Compute the sum for each phi_i
            update_terms = np.sum(opinion_diff * exp_decay, axis=1) / num_opinions
    
            # Update opinions using the formula
            new_opinions[not_convinced] = opinion_values[not_convinced] + self.beta_update * update_terms[not_convinced]
    
        # Clamp updated opinions to the range [0, 1] (just in case values exceed bounds)
        new_opinions = np.clip(new_opinions, 0, 1)
    
        # Convert the updated opinions back to a DataFrame and return it
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
        opinions = self.get_opinion(t-1)
    
        # Filter opinions to include only the relevant individuals
        opinions = opinions[opinions["id"].isin(individual_ids)].reset_index(drop=True)
    
        # Only update opinions if the activity has attendees in the specified time period
        if not opinions.empty:
    
            # Pass only the "opinion" column to the update method
            updated_opinions = self._perform_opinion_update(opinions[["opinion"]])
        
            # Update each individual's opinion history with the new opinions
            for idx, individual_id in enumerate(individual_ids):
                updated_opinion = updated_opinions.iloc[idx, 0]
                self.individuals[individual_id].opinion = updated_opinion
                self.individuals[individual_id].opinion_history[t] = updated_opinion          
      
    def run(self):
        """
        Orchestrate execution of the simulation. 
        """
        all_summaries = []  # Will hold summary of totals for each category per time period
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
                self.activities[activity].run_history.append(t)
                
            # Store activity summary at this time period
            df_activity_summary = self.activity_summary(t)
            df_activity_summary["time"] = t
            all_summaries.append(df_activity_summary)
        
        # Combine all time-period summaries into a single dataframe
        if all_summaries:
            self.overall_summary = pd.concat(all_summaries).reset_index()    
            
        self.overall_summary = (
            self.overall_summary.drop(columns="Activity") 
            .groupby("time", as_index=False)          
            .sum(numeric_only=True)                   
            )

    def plot_network(self, time):
        """
        Plot the state of the population and activities at the given time.
    
        Args:
            time (int): The time point at which to plot the network.
        """
    
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Plot individuals with colors based on opinion categories
        # Extract individual positions and opinions at the specified time
        individual_positions = []
        individual_colours = []
    
        for individual in self.individuals.values():
            individual_positions.append((individual.location.x, individual.location.y))
    
            # Get opinion value at the current time
            opinion = individual.opinion_history.get(time, individual.opinion)
    
            # Assign a color based on the opinion categories
            for category, (lower, upper, colour) in self.categories.items():
                if lower <= opinion <= upper:
                    individual_colours.append(colour)  
                    break
                
        # Unpack individual positions for plotting
        x_individuals, y_individuals = zip(*individual_positions)
        ax.scatter(x_individuals, y_individuals, c=individual_colours, s=20, label="Individuals")
    
        # Plot active activities at the specified time
        activity_positions = []
        activity_lines = []  # Store lines between individuals and activities
    
        for activity_id, activity in self.activities.items():
            if time in activity.run_history:  # Only consider activities running at this time
                # Add activity position (as black triangles)
                x, y = activity.location.x, activity.location.y
                activity_positions.append((x, y))
    
                # Use get_edges_node to find individuals connected to this activity
                edges = self.op_mod_graph.get_edges_node(activity_id)
    
                # For each edge, find the individual's position and store the line
                for edge in edges:
                    individual_id = edge[0] if edge[1] == activity_id else edge[1] 
                    individual = self.individuals[individual_id]
                    activity_lines.append([(individual.location.x, individual.location.y), (x, y)])
    
        # Unpack activity positions for plotting
        if activity_positions:
            x_activities, y_activities = zip(*activity_positions)
            ax.scatter(x_activities, y_activities, c='black', marker='^', s=100, label="Activities")
    
        # Draw lines between individuals and assigned activities
        if activity_lines:
            line_collection = LineCollection(activity_lines, colors='gray', linewidths=0.5)
            ax.add_collection(line_collection)
    
        # Step 4: Finalize plot
        ax.axis("off")
        ax.set_title(f"Opinions at Time {time}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
        # Add a detailed legend for the opinion categories
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colour, markersize=8, label=category)
            for category, (_, _, colour) in Simulation.categories.items()
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label="Activities"))
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=1, label="Connections"))
        ax.legend(
            handles=legend_elements, 
            loc='best', 
            bbox_to_anchor=(1, 0.5),  
            borderaxespad=0
        )
        plt.tight_layout()
        plt.show()
    
    def chart(self):
        """
        Draws a line plot showing how the number of individuals in each opinion category 
        """
        for category, (_, _, colour) in Simulation.categories.items():
            plt.plot(
                self.overall_summary["time"], 
                self.overall_summary[category], 
                label=category, 
                color=colour
            )

        plt.xlabel("Time")
        plt.ylabel("Number")
        plt.title("Opinion Changes Over Time")
        plt.legend(title="Type")
        plt.grid(True)
        plt.show()

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
        extreme_against_range = Simulation.categories["ExtremeAgainst"]
        extreme_for_range = Simulation.categories["ExtremeFor"]
    
        max_extreme_against = 0
        max_extreme_for = 0
        max_total_extremes = 0
    
        # Identify all time steps across all individuals
        time_periods = set()
        for individual in self.individuals.values():
            time_periods.update(individual.opinion_history.keys())
        time_periods = sorted(time_periods) 
    
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
        extreme_against_range = Simulation.categories["ExtremeAgainst"]
        extreme_for_range = Simulation.categories["ExtremeFor"]
    
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
        time_periods = sorted(time_periods)  
    
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
        Summarizes opinions of individuals at time `t` into specified categories, 
        grouped by activity.
        
        Parameters:
            t (int): The time step for which opinions are being summarized.
            
        Returns:
            pd.DataFrame: A summary DataFrame grouped by activity, with counts 
            for each opinion category.
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
        """
        Generate a summary of the percentage of time each individual spends in 
        each opinion category.
        
        Returns:
            pd.DataFrame: A DataFrame with the following columns:
            - 'X': X-coordinate of the individual.
            - 'Y': Y-coordinate of the individual.
            - 'ExtremeAgainst'
            - 'ModerateAgainst'
            - 'Neutral'
            - 'ModerateFor'
            - 'ExtremeFor'
        """
        # Initialize a list to store data for each individual
        data = []
        
        # Iterate over individuals
        for individual_id, individual in self.individuals.items():
            
            # Categorize opinions for each time step
            categories = [
                Simulation._categorise_opinion(opinion)
                for opinion in individual.opinion_history.values()
            ]
            
            # Calculate the percentage of time spent in each category
            category_counts = pd.Series(categories).value_counts(normalize=True) * 100
            
            # Ensure all categories are included in the result (even if they have 0%)
            percentages = {
                f"{category}": category_counts.get(category, 0)
                for category in Simulation.categories.keys()
            }
            
            # Add the individual's data to the list
            data.append({
                "X": individual.location.x,
                "Y": individual.location.y,
                **percentages
            })
    
        # Create a DataFrame from the collected data
        return pd.DataFrame(data)
    
    def get_elbow_plot(self, t, max_clusters=10):
        """
        Generates an elbow plot for opinuions at a particular time.
        
        Args:
            t (int): Timepoint for clustering.
            max_clusters (int): Maximum number of clusters to evaluate.
        """
        # Extract opinions and locations for specified time
        data = [
            (individual.location.x, individual.location.y, individual.opinion_history[t])
            for individual in self.individuals.values()
            if t in individual.opinion_history
        ]
        data = np.array(data)  

        # Compute inertia for different numbers of clusters
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), inertia, marker="o")
        plt.xticks(range(1, max_clusters + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Distortion (inertia)")
        plt.title(f"Elbow Plot for KMeans Clustering (Time {t})")
        plt.grid(True)
        plt.show()

    def kmeans_clustering(self, t, n_clusters=4):
        """
        Performs KMeans clustering on opinions.

        Args:
            t (int): Timepoint for clustering.
            n_clusters (int): Number of clusters to use in KMeans.
        """
        
        # Extract opinions and locations at time `t`
        data = [
            (individual.location.x, individual.location.y, individual.opinion_history[t])
            for individual in self.individuals.values()
            if t in individual.opinion_history
        ]
        data = np.array(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)

        # Plot individuals in (x, y) space, colored by cluster
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            data[:, 0],  # x-coordinates
            data[:, 1],  # y-coordinates
            c=labels,  # Cluster labels as colors
            cmap="Blues",  # Color map
            s=50,  # Marker size
            alpha=0.7,  # Transparency
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"KMeans Clustering at Time {t} (k = {n_clusters})")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.show()
        
    def fit_regression_model(self, predictors=None):
        """
        Fit a linear regression model to predict final opinions based on selected independent variables.
    
        Args:
            predictors (list, optional): A list of column names to use as predictors. 
                                         Defaults to all available predictors.
    
        Returns:
            Regression model
        """
        
        # Extract data needed for regression
        df = self._get_model_data()
    
        # Default to all predictors if none are specified
        if predictors is None:
            predictors = ["InitialOpinion", "DistanceToCentre", "NearestActivityDistance"] + \
                         [col for col in df.columns if col.startswith("ActivityMembership_")]
        
        # Define independent variables (X) and dependent variable (y)
        X = df[predictors]  
        y = df["FinalOpinion"]  
        
        # Add a constant term to the model for the intercept
        X = sm.add_constant(X)
    
        # Fit the OLS regression model
        return sm.OLS(y, X).fit()
    
    def cross_validate_model(self, predictors=None):
        """
        Fit a linear regression model using training data then validate by making predictions
        using test data.
    
        Args:
            predictors (list, optional): A list of column names to use as predictors. 
                                         Defaults to all available predictors.
    
        Returns:
            Mean Square Error (MSE) of the predictions
        """
        
        # Extract data needed for regression
        df = self._get_model_data()
    
        # Default to all predictors if none are specified
        if predictors is None:
            predictors = ["InitialOpinion", "DistanceToCentre", "NearestActivityDistance"] + \
                         [col for col in df.columns if col.startswith("ActivityMembership_")]
        
        # Split data
        X = df[predictors]
        y = df["FinalOpinion"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Fit the model on training data
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()
        
        # Evaluate on testing data
        X_test = sm.add_constant(X_test)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred) 
    
    def _get_model_data(self):
        data = []   
        for individual_id, individual in self.individuals.items():      
            initial_opinion = individual.opinion_history[0]
            final_opinion = individual.opinion_history[max(individual.opinion_history.keys())]
            distance_to_centre = np.sqrt(individual.location.x**2 + individual.location.y**2)
    
            # Distance to the nearest activity
            distances_to_activities = [
                Location.distance(individual.location, activity.location)
                for activity in self.activities.values()
            ]
            nearest_activity_distance = min(distances_to_activities)
    
            # Activity memberships
            activity_memberships = [
                activity_id for activity_id, activity in self.activities.items()
                if individual_id in [edge[0] for edge in self.op_mod_graph.get_edges_node(activity_id)]
            ]
            activity_membership = activity_memberships[0] if activity_memberships else None
    
            # Append the data for this individual
            data.append({
                "InitialOpinion": initial_opinion,
                "FinalOpinion": final_opinion,
                "DistanceToCentre": distance_to_centre,
                "NearestActivityDistance": nearest_activity_distance,
                "ActivityMembership": activity_membership
            })
    
        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
    
        # Create dummy variables for activity membership
        df = pd.get_dummies(df, columns=["ActivityMembership"], drop_first=True)
        
        # Convert boolean columns to numeric (if any exist)
        bool_columns = df.select_dtypes(include="bool").columns
        df[bool_columns] = df[bool_columns].astype(int)
        
        return df

