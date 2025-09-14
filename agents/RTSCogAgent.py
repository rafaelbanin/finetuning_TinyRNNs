from .CogAgent import CogAgent


class RTSCogAgent(CogAgent):
    """Cog agents for the reduced two-step task.

    Attributes:
        config: all hyperparameters related to the agent.
        model: the Cog agent, implemented in Akam's way.
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
            'BAS': rl.BAS(),
            'RC': rl.Reward_as_cue(),
            'MFs': rl.Model_free_symm(),
            'MB0': rl.Model_based(),
            'MB1': rl.Model_based_decay(),
            'MB0s': rl.Model_based_symm(),
            'MXs': rl.Model_mixed_symm(),
            'MB0md': rl.Model_based_mix_decay(),
            'MB0m': rl.Model_based_mix(),
            'LS0': rl.Latent_state_softmax(),
            'LS1': rl.Latent_state_softmax_bias(),
            'Q(1)': rl.Q1(),
            'Q(0)': rl.Q0(),
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars

