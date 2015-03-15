import numpy as np
import scipy.stats as st
import pandas as pd
import itertools
from progressbar import ProgressBar
from enum import Enum

class Modes(Enum):
    learn_first = 0
    learn_extended = 1
    replace_ext = 2
    normal_operation = 3
class SubModes(Enum):
    new_cycle = 0
    advance_single = 1
    normal_operation = 2
    exploit = 3
    explore = 4

class MaxPeriodSolver(object):
    def __init__(self, nbandits, period, subsetsize, distribution):
        self.rng = distribution
        self.states = -1 * np.ones((nbandits, period), dtype=int)
        self.current_state = np.zeros(nbandits, dtype=int)
        self.ell = subsetsize
        
        self.mode = Modes.learn_first
        self.submode = SubModes.new_cycle
        self.phase = 0
        self.first_subset = np.arange(self.ell)
        self.extended_subset = np.arange(self.ell, self.ell + 1)
        self.current_bandit = -1
        self.current_subset = self.first_subset
        self.trial = 0
        self.inner_trial = 0
        self.last_result_vector = np.zeros(period)
        self.current_result_vector = np.zeros(period)
        
        # Flags:
        self.firstcycle = True
        
    def resolve_equals(self, bandit):
        p = self.states.shape[1]
        
        equal_spots = self.states[bandit] == -2
        if not np.all(equal_spots) and np.any(equal_spots):
            for e in np.where(equal_spots)[0]:
                i = e
                while self.states[bandit, i % p] == -2:
                    i += 1
                replace = np.arange(p).take(np.arange(e, i), mode='wrap')
                self.states[bandit, replace] = self.states[bandit, i % p]
        
    def handle_info(self):
        B = self.current_bandit
        V = self.current_result_vector
        U = self.last_result_vector
        p = self.states.shape[1]
        st = self.current_state
        ss = self.current_subset
        
        diff = V - U
        j = st[B] - 1
        for i in range(len(diff)):
            if diff[i] < 0:
                assert self.states[B, (i+j) % p] != 0
                self.states[B, (i+j) % p] = 1
                self.states[B, (i+j+1) % p] = 0
            elif diff[i] > 0:
                assert self.states[B, (i+j) % p] != 1
                self.states[B, (i+j) % p] = 0
                self.states[B, (i+j+1) % p] = 1
            else: # B[i] equals B[i+1]
                if self.states[B, (i+j) % p] == -1:
                    self.states[B, (i+j) % p] = -2
        self.resolve_equals(B)
        
                
    def single_advance_subset(self, bandit, exclude):
        K = self.states.shape[0]
        available = np.setdiff1d(np.arange(K), exclude)
        subset = available[:self.ell-1]
        return np.concatenate([[bandit], subset])
    
    def _choose_extended(self):
        p = self.states.shape[1]
        # TODO:
        # Identify easy cases:
        #  * All unknowns are equal
        #  * Phase 1 -> The new bandit was learned and is enough
        
        left2learn = np.where(np.any(self.states == -2, axis=1))[0]
        if (len(left2learn) == 0 or 
            (self.mode is Modes.replace_ext and
             not np.any(self.states[self.ell] == -2))):
            self.mode = Modes.normal_operation
            self.submode = SubModes.exploit
            return self.choose()
        
        already_learned = np.setdiff1d(self.first_subset, left2learn)
        extended = np.concatenate([self.first_subset, self.extended_subset])
        if self.mode is Modes.learn_extended:
            if not self.firstcycle and self.inner_trial % p == 0:
                if self.submode is SubModes.new_cycle:
                    self.submode = SubModes.advance_single
                    if self.current_bandit != (self.ell - 1):
                        self.handle_info()
                if self.submode is SubModes.advance_single:
                    self.current_bandit += 1
                    if self.current_bandit > self.ell:
                        self.mode = Modes.replace_ext
                        return self._choose_extended()
                    self.current_subset = self.single_advance_subset(
                        self.current_bandit,
                        self.first_subset
                    )
                    return self.current_subset
            m = len(already_learned)
            self.current_subset = np.concatenate([
                already_learned,
                left2learn[:self.ell - m - 1],
                self.extended_subset
            ])
            return self.current_subset
        else: # Replace each unkown bandit once and look for changes
            pass
        return None
    
    def generate_reality(self, left2learn):
        rewards = np.zeros(len(left2learn))
        for i, left in enumerate(left2learn):
            for s in self.states[left]:
                if s >= 0:
                    rewards[i] += s
                else:
                    rewards[i] += self.rng()
        return rewards

    def _optimistic_reality(self, bandit):
        values = np.copy(self.states[bandit])
        values[np.where(values < 0)] = 1
        return values.sum()
        
    def _choose_normal(self):
        left2learn = np.where(np.any(self.states < 0, axis=1))[0]
        learned = np.setdiff1d(np.arange(self.states.shape[0]), left2learn)
        cum_ones = self.states[learned].sum(axis=1)
        best_bandits = learned[np.argsort(cum_ones)[::-1][:self.ell]]
        worst_best = self.states[best_bandits[-1]].sum()
        if self.submode is SubModes.explore:
            self.current_subset[-1] = self.current_bandit
            st = self.states[self.current_bandit]
            if np.all(st >= 0) or self._optimistic_reality(self.current_bandit) <= worst_best:
                self.submode = SubModes.exploit
                return self._choose_normal()
            return self.current_subset
        if self.submode is SubModes.exploit:
            if len(left2learn) != 0:
                reality = self.generate_reality(left2learn)
                max_bandit = np.argmax(reality)
                if reality[max_bandit] > worst_best:
                    self.submode = SubModes.explore
                    self.current_bandit = left2learn[max_bandit]
                    return self._choose_normal()
            self.current_subset = best_bandits
            
            return best_bandits
        
    def choose(self):
        p = self.states.shape[1]
        
        if self.mode is Modes.learn_first:
            if not self.firstcycle and self.inner_trial % p == 0:
                if self.submode is SubModes.new_cycle:
                    self.submode = SubModes.advance_single
                    if self.current_bandit != -1:
                        self.handle_info()
                if self.submode is SubModes.advance_single:
                    self.current_bandit += 1
                    if self.current_bandit >= self.ell:
                        self.mode = Modes.learn_extended
                        self.current_bandit = self.ell - 1
                        self.submode = SubModes.new_cycle
                        self.firstcycle = True
                        return self.choose()
                    self.current_subset = self.single_advance_subset(
                        self.current_bandit,
                        self.first_subset
                    )
                    return self.current_subset
            self.current_subset = self.first_subset
            return self.current_subset
        elif (self.mode is Modes.learn_extended or
              self.mode is Modes.replace_ext):
            return self._choose_extended()
        else:
            return self._choose_normal()
        
    def _set_flags(self):
        p = self.states.shape[1]
        ss = self.current_subset
        
        self.current_state[ss] = (self.current_state[ss] + 1) % p
        self.firstcycle = False
        self.trial += 1
        if not self.submode is SubModes.advance_single:
            self.inner_trial += 1
            if self.inner_trial % p == 0:
                self.submode = SubModes.new_cycle
        if self.submode is SubModes.advance_single:
            self.submode = SubModes.normal_operation
    
    def reward(self, value):
        p = self.states.shape[1]
        step = self.inner_trial % p
        ss = self.current_subset
        st = self.current_state[ss]
        
        # Handle 0s and 1s here (current state information available)
        if value == 0 or value == 1:
            self.states[ss, st] = value
            for bandit in ss:
                self.resolve_equals(bandit)
        
        if (self.mode is Modes.learn_first or 
            self.mode is Modes.learn_extended):
            if self.submode is SubModes.advance_single:
                self.last_result_vector = self.current_result_vector
                self.current_result_vector = np.empty(p)
            else:
                self.current_result_vector[step] = value
            self._set_flags()
        elif self.mode is Modes.replace_ext:
            pass
        else:
            if self.submode is SubModes.explore:
                ones = int(np.around(self.ell * value))
                result = ones - self.states[ss, st][:self.ell-1].sum()
                B = self.current_bandit
                self.states[B, self.current_state[B]] = result
            self.current_state[ss] = (self.current_state[ss] + 1) % p
            self.trial += 1


class ThompsonSampling(object):
    def __init__(self, nbandits, period, subsetsize, distribution):
        self.rewards = np.zeros((nbandits, 2))
        self.ell = subsetsize
        self.rand = distribution
        self.prior = np.array([0.5, 0.5])

    def choose(self):
        a, b = (self.rewards + self.prior).T
        samples = st.beta.rvs(a, b)
        opt_i = np.argsort(samples)[::-1][:self.ell]
        self.current_subset = opt_i
        return opt_i

    def reward(self, value):
        self.rewards[self.current_subset] += (value, 1.0 - value)


class RandomSampling(object):
    def __init__(self, nbandits, period, subsetsize, distribution):
        self.bandits = np.arange(nbandits)
        self.ell = subsetsize

    def choose(self):
        return np.random.choice(self.bandits, self.ell, replace=False)

    def reward(self, value):
        pass


def opt_values(rewards, ell, nbandits, n=500):
    p = rewards.shape[1]
    bb = np.argsort(rewards.sum(axis=1))[::-1][:ell]
    st = np.zeros(nbandits, dtype=int)
    result = np.empty(n)
    for i in range(n):
        rew = rewards[bb, st[bb]]
        result[i] = rew.sum() / float(ell)
        st[bb] = (st[bb] + 1) % p
    return result

def random_values(rewards, ell, n=500):
    K, p = rewards.shape
    bandits = np.arange(K)
    st = np.zeros(nbandits, dtype=int)
    result = np.empty(n)
    for i in range(n):
        bb = np.random.choice(bandits, size=ell, replace=False)
        rew = rewards[bb, st[bb]]
        result[i] = rew.sum() / float(ell)
        st[bb] = (st[bb] + 1) % p
    return result

def inner_simulation(algorithm, trials, nbandits, nperiods,
                     subsetsize, distribution, seed, regret):
    np.random.seed(seed)
    rewards = distribution.rvs((nbandits, nperiods))
    dist = lambda : distribution.rvs()
    state = np.zeros(nbandits, dtype=int)
    alg = algorithm(nbandits, nperiods, subsetsize, dist)
    opt_rewards = opt_values(rewards, subsetsize, nbandits, n=trials)
    values = np.empty(trials)
    for t in range(trials):
        ss = alg.choose()
        rew = rewards[ss, state[ss]]
        value = rew.sum() / float(subsetsize)
        alg.reward(value)
        state[ss] = (state[ss] + 1) % nperiods
        values[t] = value
    regret[:] = np.cumsum(opt_rewards) - np.cumsum(values)
    return

def outer_simulation(trials, repetitions, parameter_steps, start_seed):
    results = [] # result dicts
    param_iter = itertools.product(
        parameter_steps['algorithm'],
        parameter_steps['nbandits'],
        parameter_steps['nperiods'],
        parameter_steps['subsetsize'],
        parameter_steps['theta']
    )
    #niter = np.prod([len(p) for p in parameter_steps.itervalues()])
    #niter *= trials * repetitions
    for a, K, p, l, th in param_iter:
        dist = st.bernoulli(th)
        regret = np.empty(trials)
        seed = start_seed
        for rep in range(repetitions):
            inner_simulation(algorithm=a, trials=trials, nbandits=K, nperiods=p,
                             subsetsize=l, distribution=dist, seed=seed, regret=regret)
            for t in range(trials):
                results.append({
                    'algorithm': a.__name__,
                    'bandits': K,
                    'periods': p,
                    'SubsetSize': l,
                    'Trial': t,
                    'Repetition': rep,
                    'theta': th,
                    'Regret': regret[t]
                })
            seed += 1
    return pd.DataFrame(results)