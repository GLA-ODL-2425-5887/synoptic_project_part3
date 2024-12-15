import pytest
import numpy as np
import pandas as pd
from simulation import Simulation  
from settings import Settings
from unittest.mock import MagicMock

def mock_settings():
    return Settings(
        t = 500,               # Duration of simulation
        n = 200,               # Number of individuals
        n_l = None,            # Locations of individuals
        alpha_pos = 0.25,      # Probability of positive individuals
        alpha_neg = 0.25,      # Probability of negative individuals
        lambda_dist = 1,       # Travel parameter
        beta_update = 0.01,    # Opinion movement parameter
        beta_spread = 0.01,    # Listening to opposing views parameter
        gamma_extr = 0.005,    # Extremism parameter
        g = (4, 4),            # 4 activity periods, each with 4 activities
        g_l = [[(0,0),(0,1),(1,0),(1,1)],[(0,0.5),(0.5,0),(1,0.5),(0.5,1)]]
    )

@pytest.fixture
def simulation_instance():
    settings = mock_settings()
    return Simulation(settings)

def test_perform_opinion_update_empty(simulation_instance):
    """Test _perform_opinion_update with an empty opinions DataFrame."""
    opinions = pd.DataFrame(columns=["opinion"])
    updated_opinions = simulation_instance._perform_opinion_update(opinions)

    assert updated_opinions.empty, "Updated opinions should be empty for empty input."
    assert list(updated_opinions.columns) == ["updated_opinion"], "Output should have the correct column name."

def test_perform_opinion_activity_valid_data(simulation_instance, mocker):
    """Test _perform_opinion_activity with valid data."""
    # Mock the relevant methods and data
    mocker.patch.object(simulation_instance, 'get_opinion', return_value=pd.DataFrame({
        "id": [1, 2, 3],
        "opinion": [0.2, 0.5, 0.7]
    }))

    mocker.patch.object(simulation_instance, '_perform_opinion_update', return_value=pd.DataFrame({
        "updated_opinion": [0.25, 0.55, 0.75]
    }))

    # Mock the graph edges
    simulation_instance.op_mod_graph.get_edges_node = MagicMock(return_value=[(1, "activity"), (2, "activity"), (3, "activity")])

    # Mock individual data
    simulation_instance.individuals = {
        1: MagicMock(opinion_history={}),
        2: MagicMock(opinion_history={}),
        3: MagicMock(opinion_history={})
    }

    # Call the method
    simulation_instance._perform_opinion_activity("activity", t=1)

    # Assert the opinions were updated
    assert simulation_instance.individuals[1].opinion == 0.25
    assert simulation_instance.individuals[2].opinion == 0.55
    assert simulation_instance.individuals[3].opinion == 0.75

    # Assert the opinion history was updated
    assert simulation_instance.individuals[1].opinion_history[1] == 0.25
    assert simulation_instance.individuals[2].opinion_history[1] == 0.55
    assert simulation_instance.individuals[3].opinion_history[1] == 0.75

def test_perform_opinion_activity_partial_match(simulation_instance, mocker):
    """Test _perform_opinion_activity with a subset of individuals attending."""
    # Mock the relevant methods and data
    mocker.patch.object(simulation_instance, 'get_opinion', return_value=pd.DataFrame({
        "id": [1, 2, 3],
        "opinion": [0.2, 0.5, 0.7]
    }))

    mocker.patch.object(simulation_instance, '_perform_opinion_update', return_value=pd.DataFrame({
        "updated_opinion": [0.25, 0.55]
    }))

    # Mock the graph edges
    simulation_instance.op_mod_graph.get_edges_node = MagicMock(return_value=[(1, "activity"), (2, "activity")])

    # Mock individual data
    simulation_instance.individuals = {
        1: MagicMock(opinion_history={}),
        2: MagicMock(opinion_history={}),
        3: MagicMock(opinion_history={})
    }

    # Call the method
    simulation_instance._perform_opinion_activity("activity", t=1)

    # Assert the opinions were updated for matched individuals
    assert simulation_instance.individuals[1].opinion == 0.25
    assert simulation_instance.individuals[2].opinion == 0.55

    # Ensure unmatched individuals are not updated
    assert 1 not in simulation_instance.individuals[3].opinion_history

    # Assert the opinion history was updated for matched individuals
    assert simulation_instance.individuals[1].opinion_history[1] == 0.25
    assert simulation_instance.individuals[2].opinion_history[1] == 0.55
