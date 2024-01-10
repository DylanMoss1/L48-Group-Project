"""Defines constant parameters for all emulator parameter spaces and defines spaces per emulator."""

from emukit.core import ParameterSpace, DiscreteParameter, ContinuousParameter

# random start day within a year
day = DiscreteParameter("day", range(1, 366, 5))
# random start temperature occurable within a year
temperature = DiscreteParameter("temperature", range(-10, 30))
# 50 x 50 grid (10 - 30% populated)
population = DiscreteParameter("population", range(250, 750, 50))
size = ContinuousParameter("size", 0, 1)
speed = ContinuousParameter("speed", 0, 1)
vision = ContinuousParameter("vision", 0, 1)
aggression = ContinuousParameter("aggression", 0, 1)
m_size = ContinuousParameter("m_size", 0, 1)
m_speed = ContinuousParameter("m_speed", 0, 1)
m_vision = ContinuousParameter("m_vision", 0, 1)
m_aggression = ContinuousParameter("m_aggression", 0, 1)
# between a day and a month
coupling = DiscreteParameter("coupling", [1, 7, 30])

coupled_space = ParameterSpace(
    [
        population,
        m_size,
        m_speed,
        m_vision,
        m_aggression,
        coupling,
    ]
)
drift_space = ParameterSpace(
    [
        temperature,
        population,
        size,
        speed,
        vision,
        aggression,
        m_size,
        m_speed,
        m_vision,
        m_aggression,
    ]
)
population_space = ParameterSpace(
    [temperature, population, size, speed, vision, aggression]
)

training_space = ParameterSpace(
    [
        day,
        population,
        size,
        speed,
        vision,
        aggression,
        m_size,
        m_speed,
        m_vision,
        m_aggression,
    ]
)
