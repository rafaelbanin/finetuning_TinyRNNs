from .CogAgent import CogAgent


class NTSCogAgent(CogAgent):
    """Cog agents for the novel two-step task.

    Attributes:
        config: all hyperparameters related to the agent.
        model: the Cog agent, most implemented in Akam's way.
        params: the parameters of the Cog agent model.
        """

    def __init__(self, config=None):
        super().__init__()
        from . import NTS_RL_agents as rl
        from . import RTS_RL_agents_v1 as rts_rl
        self.config = config
        self.cog_type = config['cog_type']

        # if 'dataset' in config and config['dataset'] != 'AkamRat':
        #     raise ValueError('NTSCogAgent only supports AkamRat dataset')
        self.model = {
            'BAS': rts_rl.BAS(),
            'Q(1)': rts_rl.Q1(),
            'MFs': rts_rl.Model_free_symm(),
            'MF_MB_bs_rb_ck': rl.MF_MB(['bs','rb','ck']),
            'MF_bs_rb_ck': rl.MF(['bs','rb','ck']),
            'MB_bs_rb_ck': rl.MB(['bs','rb','ck']),
            'MF_MB_dec_bs_rb_ck': rl.MF_MB_dec(['bs','rb','ck']),
            'MF_dec_bs_rb_ck': rl.MF_dec(['bs','rb','ck']),
            'MB_dec_bs_rb_ck': rl.MB_dec(['bs','rb','ck']),
            'MF_MB_vdec_bs_rb_ck': rl.MF_MB_vdec(['bs','rb','ck']),
            'MF_MB_bs_rb_ec': rl.MF_MB(['bs','rb','ec']),
            'MF_MB_vdec_bs_rb_ec': rl.MF_MB_vdec(['bs','rb','ec']),
            'MF_MB_dec_bs_rb_ec': rl.MF_MB_dec(['bs','rb','ec']),
            'MF_MB_dec_bs_rb_mc': rl.MF_MB_dec(['bs','rb', 'mc']),
            'MF_MB_dec_bs_rb_ec_mc': rl.MF_MB_dec(['bs','rb','ec','mc']),
            'MFmoMF_MB_dec_bs_rb_ec_mc': rl.MFmoMF_MB_dec(['bs','rb','ec','mc']),
            'MFmoMF_dec_bs_rb_ec_mc': rl.MFmoMF_dec(['bs','rb','ec','mc']),
            'MB_dec_bs_rb_ec_mc': rl.MB_dec(['bs','rb','ec','mc']),
        }[self.cog_type]
        self._set_init_params()
        if hasattr(self.model, 'state_vars'):
            self.state_vars = self.model.state_vars
