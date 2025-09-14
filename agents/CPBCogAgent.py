from .CogAgent import CogAgent


class CPBCogAgent(CogAgent):
    """Cog agents for the change-point task

    Attributes:
        config: all hyperparameters related to the agent.
        model: the Cog agent.
        params: the parameters of the Cog agent model.
        """

    def __init__(self, config=None):
        super().__init__()
        from . import RTS_RL_agents_v1 as rl
        self.config = config
        self.cog_type = config['cog_type']
        if 'dataset' in config and config['dataset'] != 'CPBHuman':
            raise ValueError('CPBCogAgent only supports CPBHuman dataset')
        self.model = {
            'MF': rl.CPB_model_free(),
            'MB': rl.CPB_model_based(),
        }[self.cog_type]
        self._set_init_params()
