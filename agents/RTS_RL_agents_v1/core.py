"""
The core architecture of the reduced two-step task.

"""
import numpy as np
from numba import jit
from random import random, randint
import sys

log_max_float = np.log(sys.float_info.max / 2.1)  # Log of largest possible floating point number. # 709.0407755486547


@jit(nopython=True)
def softmax(Q, T):
    """Softmax choice probs given values Q and inverse temp T."""
    assert len(Q.shape) == 1
    QT = Q * T
    QT -= np.max(QT)
    expQT = np.exp(QT)
    return expQT / expQT.sum()


@jit(nopython=True)
def array_softmax(Q, T):
    """
    Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.
    """
    P = Q.copy()
    TdQ = -T * (Q[:, 0] - Q[:, 1])
    TdQ[TdQ > log_max_float] = log_max_float  # Protection against overflow in exponential.
    P[:, 0] = 1. / (1. + np.exp(TdQ))
    P[:, 1] = 1. - P[:, 0]
    return P


@jit(nopython=True)
def protected_log(x):
    """Return log of x protected against giving -inf for very small values of x."""
    return np.log(((1e-200) / 2) + (1 - (1e-200)) * x)


@jit(nopython=True)
def _compute_loglik(choice_probs, c):
    if isinstance(c, (int, np.integer, np.uint8, np.int32)): # c is single choice
        trial_log_likelihood = protected_log(choice_probs[c])
    else: # c is an array of choice probabilities
        assert c.shape[0] == choice_probs.shape[0]
        trial_log_likelihood = np.sum(protected_log(choice_probs) * c)
    return trial_log_likelihood

@jit(nopython=True)
def _compute_kl(choice_probs, target):
    """Compute KL divergence between choice_probs and target."""
    return np.sum(target * (np.log(target + 1e-200) - np.log(choice_probs + 1e-200)))
@jit(nopython=True)
def _step_other_variables(iTemp, Q):
    scores = Q * iTemp
    choice_probs = softmax(scores, 1)
    return scores, choice_probs


def choose(P):
    """Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"""
    return sum(np.cumsum(P) < random())


class TwoStepModelCore():
    """Core of the two-step model.

    Every concrete two-step model should inherit from this class,
    and implement the following methods:
        _init_core_variables
        _step_core_variables
        _step_other_variables
        _session_likelihood_core
    _init_core_variables and _step_other_variables are used to initialize the working memory.
    _step_core_variables and _step_other_variables are used to update the working memory, called by step_wm() in simulate().
    _session_likelihood_core is used to compute the session likelihood, called by session_likelihood().
    _step_core_variables, _step_other_variables, _session_likelihood_core can call functions optimized by numba.
    """

    def __init__(self):
        self.wm = None
        self.params = []
        self.state_vars = []

    def _pack_trial_event(self, c, obs):
        """Return a tuple of trial event. See TwoStepModelCoreCSO for details."""
        raise NotImplementedError

    def _update_session(self, session, n_trials, trial, trial_event):
        """Update session data given trial event. See TwoStepModelCoreCSO for details."""
        raise NotImplementedError

    def _init_core_variables(self, wm, params):
        """Initialize core variables before the first trial.

        Args:
            wm: Working memory, a dictionary. If None, initialize a new one.
            params (list): Model parameters. Cannot be None.
        """
        raise NotImplementedError

    def _step_core_variables(self, trial_event, params):
        """Step core variables (usually Q values before the scores) given trial event and parameters.

        Args:
            trial_event (tuple): Trial event.
            params (list): Model parameters. Cannot be None.
            """
        raise NotImplementedError

    def _step_other_variables(self, params):
        """Step other variables (the scores and choice_probs) given parameters.

        Args:
            params (list): Model parameters. Cannot be None.
        """
        raise NotImplementedError

    def _session_likelihood_core(self, session, params, DVs):
        """Core function for computing session likelihood.

        This function assumes that the initial wm is already saved in DVs for trial=0.
        Then DVs is calculated for all trials.
        Will be called by session_likelihood().
        Args:
            session (dict): Session data, usually contains choices, rewards, states, etc.
            params (list): Model parameters. Cannot be None.
            DVs (dict): Dictionary of decision variables, each of which is a numpy array of shape (n_trials + 1, ...).
        Returns:
            DVs (dict): Updated decision variables after the session.
            """

        raise NotImplementedError

    def init_wm(self, wm=None, params=None, **kwargs):
        """Initialize working memory before any task observations."""
        if params is None:
            params = self.params
        self._init_core_variables(wm, params)
        self._step_other_variables(params)

    def step_wm(self, trial_event, params=None):
        """Step working memory given trial event."""
        if params is None:
            params = self.params
        self._step_core_variables(trial_event, params)
        self._step_other_variables(params)

    def delete_wm(self):
        self.wm = None

    def simulate(self, task, n_trials, params=None, get_DVs=False):
        raise NotImplementedError

    def session_likelihood(self, session, params=None, get_DVs=False):
        """Return log likelihood of session given model parameters.

        Args:
            session: Session data. Access attributes by session['attr'] (standard format) or session.CSTO['attr'] (old format).
                n_trials (int): Number of trials in session.
                choices (numpy array): Choices in session.
                second_steps (numpy array): Second steps in session.
                outcomes (numpy array): Outcomes in session.
            params (list): Model parameters.
            get_DVs (bool): Whether to return decision variables.
        """
        if self.wm is None:
            self.init_wm(params=params)  # else, assume wm is already initialized
        if params is None:
            params = self.params
        mask = None
        if isinstance(session, dict):  # standard format
            n_trials = session['n_trials']
            if 'mask' in session: mask = session['mask']
        else:  # old cog_session format
            n_trials = session.n_trials
            session = session.CTSO
            session['n_trials'] = n_trials
        DVs = {}
        for k, v in self.wm.items():
            DVs[k] = np.zeros((n_trials + 1,) + v.shape)  # +1 for initial state
            DVs[k][0] = v  # initial wm state before the first trial
        DVs = self._session_likelihood_core(session, params, DVs)
        if mask is None:
            session_log_likelihood = np.sum(DVs['trial_log_likelihood'])
        else:
            session_log_likelihood = np.sum(DVs['trial_log_likelihood'] * mask.reshape(DVs['trial_log_likelihood'].shape))
        self.delete_wm()

        if get_DVs:
            return DVs | {'session_log_likelihood': session_log_likelihood}
        else:
            return session_log_likelihood


class TwoStepModelCoreCSO(TwoStepModelCore):
    """Core functions for two-step models with choice, second step, and outcome."""

    def _pack_trial_event(self, c, obs):
        """Pack trial event.

        Args:
            c (int): Choice.
            obs (tuple): Observations, usually (s, o).
        """
        return (c,) + obs

    def _update_session(self, session, n_trials, trial, trial_event):
        """Update session data.

        Args:
            session (dict): Session data.
            n_trials (int): Number of trials in session.
            trial (int): Trial number.
            trial_event (tuple): Trial event.
        """
        session.setdefault('choices', np.zeros(n_trials, dtype=int))[trial] = trial_event[0]
        session.setdefault('second_steps', np.zeros(n_trials, dtype=int))[trial] = trial_event[1]
        session.setdefault('outcomes', np.zeros(n_trials, dtype=int))[trial] = trial_event[2]

    def simulate(self, task, n_trials, params=None, get_DVs=False
                 ):
        """Interact with task. Return session data and decision variables.  """
        if self.wm is None:
            self.init_wm(params=params)  # else, assume wm is already initialized
        if params is None:
            params = self.params

        DVs = {}
        for k, v in self.wm.items():
            DVs[k] = np.zeros((n_trials + 1,) + v.shape)  # +1 for initial state
            DVs[k][0] = v  # initial wm state before the first trial

        session = {'n_trials': n_trials}

        task.reset(n_trials)

        for trial in range(n_trials):
            # Generate the trial event.
            c = choose(self.wm['choice_probs'])
            obs = task.trial(c)  # usually (s, o)
            trial_event = self._pack_trial_event(c, obs)  # usually (c, s, o)
            self.step_wm(trial_event, params)
            for k, v in self.wm.items():
                DVs[k][trial + 1] = v
            self._update_session(session, n_trials, trial, trial_event)

        DVs['trial_log_likelihood'] = protected_log(DVs['choice_probs'][np.arange(n_trials), session['choices']])
        session_log_likelihood = np.sum(DVs['trial_log_likelihood'])
        self.delete_wm()

        if get_DVs:
            return DVs | {'session_log_likelihood': session_log_likelihood} | session
        else:
            return session_log_likelihood

class TwoStepModelCoreOri(TwoStepModelCore):
    """Core functions for original two-step models,
    with first-stage choice, second-stage state,
    second-stage choice and outcome."""

    def _pack_trial_event(self, c, obs):
        # used in simulate()
        raise NotImplementedError

    def _update_session(self, session, n_trials, trial, trial_event):
        # used in simulate()
        raise NotImplementedError

    def simulate(self, task, n_trials, params=None, get_DVs=False):
        raise NotImplementedError
