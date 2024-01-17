"""Defines constant parameters for all emulator parameter spaces and defines spaces per emulator."""

from emukit.core import ParameterSpace, DiscreteParameter, ContinuousParameter

# random start day within a year
day = DiscreteParameter("day", range(1, 301, 10))
# random start temperature occurable within a year
temperature = DiscreteParameter("temp.", range(-10, 30))
# 50 x 50 grid (10 - 30% populated)
population = DiscreteParameter("pop.", range(250, 750, 50))
size = ContinuousParameter("sz", 0, 1)
speed = ContinuousParameter("spd", 0, 1)
vision = ContinuousParameter("vsn", 0, 1)
aggression = ContinuousParameter("agg", 0, 1)
var_size = ContinuousParameter("var_sz", 0, 0.5)
var_speed = ContinuousParameter("var_spd", 0, 0.5)
var_vision = ContinuousParameter("var_vsn", 0, 0.5)
var_aggression = ContinuousParameter("var_agg", 0, 0.5)
m_size = ContinuousParameter("m_sz", 0, 1)
m_speed = ContinuousParameter("m_spd", 0, 1)
m_vision = ContinuousParameter("m_vsn", 0, 1)
m_aggression = ContinuousParameter("m_agg", 0, 1)
output = DiscreteParameter("out_i", [0, 1, 2, 3])
# between a day and a month
coupling = DiscreteParameter("cpling", [1, 7, 30])

test_space = ParameterSpace([m_size, m_speed, m_vision, m_aggression])

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
        var_size,
        var_speed,
        var_vision,
        var_aggression,
        m_size,
        m_speed,
        m_vision,
        m_aggression,
        output,
    ]
)
population_space = ParameterSpace(
    [
        temperature,
        population,
        size,
        speed,
        vision,
        aggression,
        var_size,
        var_speed,
        var_vision,
        var_aggression,
    ]
)

training_space = ParameterSpace(
    [
        day,
        population,
        size,
        speed,
        vision,
        aggression,
        var_size,
        var_speed,
        var_vision,
        var_aggression,
        m_size,
        m_speed,
        m_vision,
        m_aggression,
    ]
)
