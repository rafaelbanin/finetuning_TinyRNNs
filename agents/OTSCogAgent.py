from .CogAgent import CogAgent


class OTSCogAgent(CogAgent):
    """Cog agents for the original two-step task.

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
        # if 'dataset' in config and config['dataset'] != 'MillerRat':
        #     raise ValueError('RTSCogAgent only supports MillerRat dataset')
        self.model = {
            # 'BAS': rl.BAS(),
            'MF': rl.Origin_model_based(model_type='MF'),
            'MB': rl.Origin_model_based(model_type='MB'),
            'MX': rl.Origin_model_based(model_type='MX'),
            'MFs': rl.Origin_model_based_symm(model_type='MF'),
            'MBs': rl.Origin_model_based_symm(model_type='MB'),
            'MXs': rl.Origin_model_based_symm(model_type='MX'),
            'MFsr': rl.Origin_model_free_symm_rew(),
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars

