from .CogAgent import CogAgent


class PRLCogAgent(CogAgent):
    """Cog agents for the probabilistic probability reversal task.

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
        # if 'dataset' in config and config['dataset'] != 'BartoloMonkey':
        #     raise ValueError('PRLCogAgent only supports BartoloMonkey dataset')
        self.model = {
            'BAS': rl.BAS(),
            'RC': rl.Reward_as_cue(),
            'MB0': rl.Model_based(p_transit=1),
            'MBf': rl.Model_based_forgetful(p_transit=1),
            'MB0off': rl.Model_based_offset(p_transit=1),
            'MB0s': rl.Model_based_symm(p_transit=1),
            'MF0sp': rl.Model_free_symm_pers(),
            'MB0se': rl.Model_based_symm(p_transit=1, equal_reward=True),
            'MB0md': rl.Model_based_mix_decay(p_transit=1),
            'MB0mdnb': rl.Model_based_mix_decay(p_transit=1, b=0),
            'MB0m': rl.Model_based_mix(p_transit=1),
            'MB1': rl.Model_based_decay(p_transit=1),
            'MFDp': rl.Model_free_decay_pers(n_actions=2),
            'MB0p': rl.Model_based_decay_pers(p_transit=1, use_decay=False),
            'MB1p': rl.Model_based_decay_pers(p_transit=1, use_decay=True),
            'LS0': rl.Latent_state_softmax(good_prob=0.7),
            'LS1': rl.Latent_state_softmax_bias(good_prob=0.7),
            'Q(0)': rl.Q0(),
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars


