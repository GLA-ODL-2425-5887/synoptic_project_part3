import pytest
from simulation import Simulation
from opinion_model.individual import NegativeIndividual, PositiveIndividual, UnbiasedIndividual
from opinion_model.location import Location
from settings import Settings
from collections import Counter
import numpy as np
import copy


@pytest.fixture
def valid_settings():
    """Fixture to create valid Settings instance."""
    return Settings(
        t=500,               # Duration of simulation
        n=200,               # Number of individuals
        n_l=None,            # Locations of individuals
        alpha_pos=0.25,      # Probability of positive individuals
        alpha_neg=0.25,      # Probability of negative individuals
        lambda_dist=1,       # Travel parameter
        beta_update=0.01,    # Opinion movement parameter
        beta_spread=0.01,    # Listening to opposing views parameter
        gamma_extr=0.005,    # Extremism parameter
        g=(4, 4),            # 4 activity periods, each with 4 activities
        g_l=[[(0,0),(0,1),(1,0),(1,1)],[(0,0.5),(0.5,0),(1,0.5),(0.5,1)]]
    )

@pytest.fixture
def custom_settings(valid_settings):
    """Fixture for creating custom settings."""
    settings_copy = copy.deepcopy(valid_settings)  # Always work on a copy
    return settings_copy

def test_simulation_initialisation_1(valid_settings):
    """Test basic initialisation of the Simulation class."""
    sim = Simulation(valid_settings)
    
    # Test individuals are correctly created
    assert len(sim.individuals) == valid_settings.n
    assert len(sim.op_mod_graph.get_nodes(0)) == valid_settings.n

def test_simulation_initialisation_2(valid_settings):
    """Test initialisation of individuals."""
    sim = Simulation(valid_settings)
    
    # Check that individuals are of the correct types
    for individual in sim.individuals.values():
        assert isinstance(individual, (NegativeIndividual, PositiveIndividual, UnbiasedIndividual))
    
def test_simulation_initialisation_3(valid_settings):
    """Test initialisation of the activity periods and activities."""
    sim = Simulation(valid_settings)
    
    # Test the number of activity periods and activities
    for period_index in range(1, valid_settings.g[0] + 1):
        activity_nodes = sim.op_mod_graph.get_nodes(group=period_index)
        assert len(activity_nodes) == valid_settings.g[1]
    
    # Ensure activities have correct locations
    assert len(sim.activities) == valid_settings.g[0] * valid_settings.g[1]
    for location in sim.activities.values():
        assert isinstance(location, Location)
        assert 0 <= location.x <= 1
        assert 0 <= location.y <= 1

def test_alpha_probabilities(valid_settings):
    """Test that the proportion of individual types matches their probabilities."""
    sim = Simulation(valid_settings)
    
    # Use type names for counting
    counts = Counter(type(ind).__name__ for ind in sim.individuals.values())
    
    total = sum(counts.values())
    assert total == valid_settings.n
    
    print(f"Counts: {counts}")
    print(f"Proportions: {counts['PositiveIndividual'] / total}, "
              f"{counts['NegativeIndividual'] / total}, "
              f"{counts['UnbiasedIndividual'] / total}")
    
    # Validate proportions (allowing some deviation due to randomness)
    assert counts["PositiveIndividual"] / total == pytest.approx(valid_settings.alpha_pos, rel=0.2)
    assert counts["NegativeIndividual"] / total == pytest.approx(valid_settings.alpha_neg, rel=0.2)
    assert counts["UnbiasedIndividual"] / total == pytest.approx(1 - valid_settings.alpha_pos - valid_settings.alpha_neg, rel=0.2)

def test_compute_activity_probabilities_single_activity(custom_settings):
    """Test probabilities calculation with a single activity."""
    custom_settings.g = (4, 1)
    sim = Simulation(custom_settings)
    
    x, y = 0.5, 0.5  # Individual's location

    # Retrieve the dynamically generated activity ID
    group = 2  # Activity group
    activity_ids = sim.op_mod_graph.get_nodes(group)

    # Ensure there is only one activity (mock Settings if needed)
    assert len(activity_ids) == 1
    activity_id = activity_ids[0]

    # Compute probabilities
    activity_probabilities = sim._compute_activity_probabilities(x, y, group)

    # Validate that the result is a list with a single tuple
    assert len(activity_probabilities) == 1
    assert activity_probabilities[0][0] == activity_id
    assert activity_probabilities[0][1] == pytest.approx(1.0)


def test_compute_activity_probabilities_equal_distances(valid_settings):
    """Test that probabilities are equal when activities are equidistant."""
    sim = Simulation(valid_settings)
    
    x, y = 0.5, 0.5  # Individual's location
    group = 2  # Activity group

    # Retrieve the activity IDs and ensure at least two are available
    activity_ids = sim.op_mod_graph.get_nodes(group)
    assert len(activity_ids) >= 2, "The simulation must have at least two activities for this test."

    # Manually set all activity locations to be equidistant from (x, y)
    equidistant_location = Location(0.5, 0.5)
    for activity_id in activity_ids:
        sim.activities[activity_id] = equidistant_location

    # Compute probabilities
    activity_probabilities = sim._compute_activity_probabilities(x, y, group)

    # Validate that all activities have equal probabilities
    expected_prob = 1 / len(activity_ids)
    for _, prob in activity_probabilities:
        assert prob == pytest.approx(expected_prob, rel=1e-3)


def test_compute_activity_probabilities_varying_distances(valid_settings):
    """Test probabilities when activities are at varying distances."""
    sim = Simulation(valid_settings)
    
    x, y = 0.5, 0.5  # Individual's location
    group = 2  # Activity group

    # Retrieve the activity IDs and ensure there are at least three activities
    activity_ids = sim.op_mod_graph.get_nodes(group)
    assert len(activity_ids) >= 3, "The simulation must have at least three activities for this test."

    # Set activity locations to different distances from (x, y)
    locations = [Location(0.6, 0.6), Location(0.8, 0.8), Location(0.2, 0.2)]
    for activity_id, location in zip(activity_ids, locations):
        sim.activities[activity_id] = location

    # Compute probabilities
    activity_probabilities = sim._compute_activity_probabilities(x, y, group)

    # Extract probabilities and activity IDs
    sorted_ids, sorted_probs = zip(*activity_probabilities)

    # Expect the closest activity to have the highest probability
    distances = [Location.distance(Location(x, y), loc) for loc in locations]
    closest_index = np.argmin(distances)

    assert sorted_ids[0] == activity_ids[closest_index]  # Closest activity should be first
    assert sorted_probs[0] > sorted_probs[-1]  # Closest activity has the highest probability
    assert sum(sorted_probs) == pytest.approx(1.0, rel=1e-3)
    
def test_individuals_assigned_to_activities(valid_settings):
    """Test that each individual is assigned to one activity in each period."""
    sim = Simulation(valid_settings)

    # Verify the structure of the graph: all individuals are connected to one activity per period
    for individual_id in sim.op_mod_graph.get_nodes(group=0):
        for period_index in range(1, sim.g[0] + 1):
            edges = [edge for edge in sim.op_mod_graph.get_edges_group(group1=period_index) 
                     if individual_id in edge]
            assert len(edges) == 1, f"Individual {individual_id} should be assigned to exactly one activity in period {period_index}."

def test_activity_selection_probabilities(valid_settings):
    """Test that activity selection respects relative probabilities."""
    sim = Simulation(valid_settings)

    x, y = 0.5, 0.5  # Arbitrary location for testing
    period_index = 1  # Test with the first activity period
    activity_probabilities = sim._compute_activity_probabilities(x, y, period_index)

    activities, probs = zip(*activity_probabilities)

    # Simulate many allocations to estimate actual selection probabilities
    counts = {activity: 0 for activity in activities}
    n_trials = 1000

    for _ in range(n_trials):
        selected_activity = np.random.choice(activities, p=probs)
        counts[selected_activity] += 1

    # Compare observed selection proportions to expected probabilities
    observed_probs = np.array(list(counts.values())) / n_trials
    assert observed_probs == pytest.approx(probs, rel=0.2), \
        "Observed selection probabilities do not match expected probabilities."

def test_individual_edges_consistency(valid_settings):
    """Test that edges are correctly created between individuals and selected activities."""
    sim = Simulation(valid_settings)

    for individual_id in sim.op_mod_graph.get_nodes(group=0):
        for period_index in range(1, sim.g[0] + 1):
            edges = [edge for edge in sim.op_mod_graph.get_edges_group(group1=period_index) 
                     if individual_id in edge]
            print(f"Individual {individual_id} Period {period_index} Edges {edges}")
            assert len(edges) == 1, \
                f"Individual {individual_id} should have exactly one edge to an activity in period {period_index}."
            activity_id = edges[0][1]
            assert activity_id in sim.op_mod_graph.get_nodes(group=period_index), \
                f"Activity {activity_id} must exist in period {period_index}."

def test_randomness_in_activity_selection(valid_settings):
    """Test that randomness in activity selection produces variability."""
    sim = Simulation(valid_settings)

    x, y = 0.5, 0.5  # Arbitrary location for testing
    period_index = 1  # Test with the first activity period
    activity_probabilities = sim._compute_activity_probabilities(x, y, period_index)

    activities, probs = zip(*activity_probabilities)

    # Perform multiple runs to observe variability in selected activities
    selected_activities = [
        np.random.choice(activities, p=probs) for _ in range(100)
    ]
    unique_selected = set(selected_activities)

    assert len(unique_selected) > 1, \
        "Random sampling should produce variability in selected activities."