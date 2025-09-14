from .CogAgent import CogAgent


class MABCogAgent(CogAgent):
    """Cog agents for the multi-arm bandit task.

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
        self.model = {
            # 'BAS': rl.BAS(),
            'MF': rl.Model_free_decay(n_actions=config['n_actions'], decay=False), # model-free without decay
            'MFD': rl.Model_free_decay(n_actions=config['n_actions']), # model-free with decay
            'MFDp': rl.Model_free_decay_pers(n_actions=config['n_actions']), # model-free with decay and perseveration
            'MFL': rl.Model_free_learn(n_actions=config['n_actions']), # model-free with learning all chosen/unchosen actions, alpha and beta are tied, continuous rewards
            'MFLabs': rl.Model_free_learn(n_actions=config['n_actions'], alpha_beta_sep=True), # model-free with learning chosen/unchosen actions, alpha and beta are separated, continuous rewards
            'MFLb': rl.Model_free_learn_all_binary(n_actions=config['n_actions']), # model-free with learning all chosen/unchosen actions, binary rewards
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars

