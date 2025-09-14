"""Functions for creating agents from a string."""
from .RNNAgent import RNNAgent, _tensor_structure_to_numpy
from .CogAgent import CogAgent
from .PRLCogAgent import PRLCogAgent
from .RTSCogAgent import RTSCogAgent
from .NTSCogAgent import NTSCogAgent
from .RNNAgentTrainer import RNNAgentTrainer
from .CogAgentTrainer import CogAgentTrainer
import importlib


def Trainer(ag):
    """Get trainer instance from agent instance.
    Each agent type has a corresponding trainer wrapper.
    """
    if isinstance(ag, RNNAgent):
        return RNNAgentTrainer(ag)
    elif isinstance(ag, CogAgent):
        return CogAgentTrainer(ag)
    else:
        raise ValueError

def Analyzer(ag):
    raise NotImplementedError

def Agent(agent_type, pipeline=None, *args, ** kwargs):
    """Get agent instance from string.

    Args:
        agent_type: A string.
        pipeline: A list of wrapper operations to be performed on the agent.
    Returns:
        A instance of desired agent with specified config.
    """
    if pipeline is None:
        pipeline = []
    # first load the agent file/module, then get the agent class, and return an instance of the class
    agent_class_name = agent_type + 'Agent'
    agent_type_class = importlib.import_module('.' + agent_class_name, package='agents')
    agent_class = getattr(agent_type_class, agent_class_name)
    ag = agent_class(*args, **kwargs)

    for op in pipeline:
        if op == 'train':
            ag = Trainer(ag)
        elif op == 'analyze':
            ag = Analyzer(ag)
        else:
            raise ValueError
    return ag