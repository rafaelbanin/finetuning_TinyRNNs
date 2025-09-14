import numpy as np
from numba import jit
import sys

from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest floating point value.

def log_safe(x):
    '''Return log of x protected against giving -inf for very small values of x.'''
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

@jit
def exp_mov_ave(data, tau = 8., initValue = 0., alpha = None):
    '''Exponential Moving average for 1d data.  The decay of the exponential can
    either be specified with a time constant tau or a learning rate alpha.'''
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(data.shape[0]+1)
    mov_ave[0] = initValue
    data_float = data.astype(float)
    for i, x in enumerate(data_float):
        mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x
    return mov_ave[1:]


class _exp_mov_ave:
    'Exponential moving average class.'
    def __init__(self, tau=None, init_value=0., alpha = None):
        if alpha is None: alpha = 1 - np.exp(-1/tau)
        self._alpha = alpha
        self._m = 1 - alpha
        self.init_value = init_value
        self.reset()

    def reset(self, init_value = None):
        if init_value:
            self.init_value = init_value
        self.ave = self.init_value

    def update(self, sample):
        self.ave = (self.ave*self._m) + (self._alpha*sample)


def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > log_max_float] = log_max_float # Protection agairt overflow in exponential.
    expQT = np.exp(QT)
    return expQT/expQT.sum()

def array_softmax(Q,T = None):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([2,n_trials])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    if T is None: # Use default temperature of 1.
        TdQ = Q[1,:]-Q[0,:]
    else: 
        TdQ = T*(Q[1,:]-Q[0,:])
    TdQ[TdQ > log_max_float] = log_max_float # Protection agairt overflow in exponential.
    P[0,:] = 1./(1. + np.exp(TdQ))
    P[1,:] = 1. - P[0,:]
    return P

def choose(P):
    "Takes vector of probabilities P summing to 1, returr integer s with prob P[s]"
    return sum(np.cumsum(P)<np.random.rand(1))

def session_log_likelihood(choices, Q_net, iTemp = None):
    'Evaluate session log likelihood given choices, action values and softmax temp.'
    choice_probs = array_softmax(Q_net, iTemp)
    session_log_likelihood = np.sum(log_safe(
                                choice_probs[choices,np.arange(len(choices))]))
    return session_log_likelihood

@jit
def _mc_kernel(choices, second_steps, alpMC):
    '''Evaluate modifier to Q values due to seperate exponential choice kernels
    for choices following second step states A and B.'''
    kq = np.zeros([2, len(choices)])
    ps = 0 #Previous trial second step
    for i, (c, s) in enumerate(zip(choices[:-1], second_steps)):
        pr = 1 - ps
        kq[pr,i+1] = kq[pr,i]
        kq[ps,i+1] = (1.-alpMC)*kq[ps,i]+alpMC*(c-0.5)
        ps = s
    kq = kq[second_steps[:-1],np.arange(1,len(choices))]
    return kq

# -------------------------------------------------------------------------------------
# Base class
# -------------------------------------------------------------------------------------

class RL_agent:

    def __init__(self, kernels = None):
        if kernels:
            self.use_kernels = True
            self.name = self.name + ''.join(['_'+k for k in kernels])
            for k in kernels:
                if k in ['bs', 'ec', 'mc', 'ck', 'rb']:
                    self.param_names  += [k]
                    self.param_ranges += ['unc']
                    if k == 'ec':
                        self.param_names  += ['alpEC']
                        self.param_ranges += ['unit' ]
                    if k == 'mc':
                        self.param_names  += ['alpMC']
                        self.param_ranges += ['unit' ]
                else:
                    assert False, 'Kernel type not recognised.'
        else:
            self.use_kernels = False
        self.n_params = len(self.param_names)
        self.calculates_gradient = False
        self.type = 'RL'

    def apply_kernels(self, Q_pre, choices, second_steps, params_T):
        '''Apply modifier to entire sessions Q values due to kernels. 
        Kernel types:
        bs - Bias high vs low.
        rb - Rotational bias (clockwise vs counterclockwise).
        ck - Choice kernel.
        ec - Exponentially decaying choice kernel.
        mc - Exponentially decaying motor level choice kernel.'''
        if not self.use_kernels: return Q_pre                
        p_names = self.param_names
        bias = params_T[p_names.index('bs')] if 'bs' in p_names else 0.
        ck   = params_T[p_names.index('ck')] if 'ck' in p_names else 0.
        rb   = params_T[p_names.index('rb')] if 'rb' in p_names else 0.
        kernel_Qs = np.zeros((2,len(choices)))
        kernel_Qs[1, :] += bias
        kernel_Qs[1,1:] += ck*(choices[:-1]-0.5)+rb*(second_steps[:-1]-0.5)
        if 'ec' in p_names:
            alpEC = params_T[p_names.index('alpEC')]
            ec   = params_T[p_names.index('ec')]
            kernel_Qs[1,1:] += ec*exp_mov_ave(choices-0.5, alpha=alpEC)[:-1]
        if 'mc' in p_names:
            alpMC = params_T[p_names.index('alpMC')]
            sck   = params_T[p_names.index('mc')]
            kernel_Qs[1,1:] += sck*_mc_kernel(choices, second_steps, alpMC)
        return Q_pre + kernel_Qs

    def init_kernels_sim(self, params_T):
        # Initialise kernels at start of simulation run.
        p_names = self.param_names
        bias = params_T[p_names.index('bs')] if 'bs' in p_names else 0.
        ck   = params_T[p_names.index('ck')] if 'ck' in p_names else 0.
        rb   = params_T[p_names.index('rb')] if 'rb' in p_names else 0.
        ec   = params_T[p_names.index('ec')] if 'ec' in p_names else 0.
        sck  = params_T[p_names.index('mc')] if 'mc' in p_names else 0.
        self.kernel_param_values = (bias,ck,rb,ec,sck)
        if 'ec' in p_names:
            alpEC = params_T[p_names.index('alpEC')]
            self.ec_mov_ave = _exp_mov_ave(alpha=alpEC)
        if 'mc' in p_names:
            alpMC = params_T[p_names.index('alpMC')]
            self.mc_mov_ave = [_exp_mov_ave(alpha=alpMC),_exp_mov_ave(alpha=alpMC)]
        self.prev_second_step = 0


    def apply_kernels_sim(self, Q_pre, c, s):
        ''' Evaluate modifier to action values due to kernels for single trial, called 
        on each trials of simulation run.'''
        if not self.use_kernels: return Q_pre   
        bias,ck,rb,ec,sck = self.kernel_param_values
        kernel_Qs = np.zeros(2)
        kernel_Qs[1] = bias + ck*(c-0.5)+rb*(s-0.5)
        if ec:
            self.ec_mov_ave.update(c-0.5)
            kernel_Qs[1] += self.ec_mov_ave.ave*ec
        if sck:
            self.mc_mov_ave[self.prev_second_step].update(c-0.5)
            kernel_Qs[1] += self.mc_mov_ave[s].ave*sck
            self.prev_second_step = s
        return Q_pre + kernel_Qs


    def get_DVs(self, session, params_T, Q_net, Q_td=None, Q_mb=None, Q_tdm=None):
        'Make dictionary containing trial-by-trial values of decision variables.'
        p = dict(zip(self.param_names, params_T)) # Parameter dictionary.
        iTemp = 1. if not 'iTemp' in p.keys() else p['iTemp']
        DVs = {'Q_net'       : Q_net,
               'choice_probs': array_softmax(Q_net, iTemp)}
        DVs['P_mf'] = np.zeros(Q_net.shape[1])
        if Q_td is not None:
            G_td  = 1. if not 'G_td'  in p.keys() else p['G_td']
            DVs['Q_td'] = Q_td
            DVs['P_td'] = iTemp * G_td * (Q_td[1,:] - Q_td[0,:])
            DVs['P_mf'] += DVs['P_td']
        if Q_tdm is not None:
            G_tdm = 1. if not 'G_tdm' in p.keys() else p['G_tdm']
            DVs['Q_tdm'] = Q_tdm
            DVs['P_tdm'] = iTemp * G_tdm * (Q_tdm[1,:] - Q_tdm[0,:])
            DVs['P_mf'] += DVs['P_tdm']
        if Q_mb is not None:
            G_mb  = 1. if not 'G_mb'  in p.keys() else p['G_mb']
            DVs['Q_mb'] = Q_mb
            DVs['P_mb'] = iTemp * G_mb * (Q_mb[1,:]  - Q_mb[0,:])
        if self.use_kernels: 
            choices, second_steps = session.unpack_trial_data('CS')
            DVs['P_k'] = self.apply_kernels(Q_net, choices, second_steps,
                                            params_T)[0,:] - Q_net[0,:]
        return DVs