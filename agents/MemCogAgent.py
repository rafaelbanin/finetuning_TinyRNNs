from .CogAgent import CogAgent


class MemCogAgent(CogAgent):
    """Cog agents for the Collins' memory task.

    Attributes:
        config: all hyperparameters related to the agent.
        model: the Cog agent.
        params: the parameters of the Cog agent model.
        """

    def __init__(self, config=None):
        super().__init__()
        from . import Mem_agents as ag
        self.config = config
        self.cog_type = config['cog_type']
        if 'dataset' in config and config['dataset'] != 'LaiHuman':
            raise ValueError('MemCogAgent only supports LaiHuman dataset')
        self.model = {
            'AC3': ag.Actor_critic(update_p=False, update_beta=False),
            'AC4': ag.Actor_critic(update_p=True, update_beta=False),
            'AC5': ag.Actor_critic(update_p=False, update_beta=True),
            'AC6': ag.Actor_critic(update_p=True, update_beta=True),
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars
