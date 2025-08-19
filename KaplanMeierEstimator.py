#
import sys
import pickle
import numpy as np
import pandas as pd

class KaplanMeierEstimator:
    '''# Object of the Kaplan-Meier estimator.'''
    def __init__(self, T = None, C = None, path_to_load=None):
        '''Constructor of the object.'''
        # load fitted estimator
        if path_to_load is not None:
            self.load(path_to_load)
        # build new estimator based on T and C
        else:
            T = list(T)
            C = list(C)
            if len(T) == len(C) and len(T) > 0:
                self.build(T, C)
            else:
                raise ValueError('ERROR! len(T) is not equal to len(C).')

    # строим новую оценку
    def build(self, T, C):
        '''Method to build a new Kaplan-Meier estimator.'''
        # initialization
        n = len(T)
        self.X = [0.0] * n
        self.F = [0.0] * n

        # sort values
        df = pd.DataFrame()
        df['T'] = list(T)
        df['C'] = list(C)
        df.sort_values(by=['T', 'C'], ascending=(True, True), inplace=True)
        T = df['T'].to_list()
        C = df['C'].to_list()

        # fix the last observation
        self.last_is_censored = True if C[-1] == 1 else False

        # compute values of the estimator
        val = 1.0
        for i in range(n):
            self.X[i] = T[i]
            if C[i] == 0:
                self.F[i] = val * (n - i - 1) / (n - i)
                val = self.F[i]
            else:
                self.F[i] = val

        # resulting form
        for i in range(n - 1, 0, -1):
            self.F[i] = self.F[i-1]
        self.F[0] = 1.0

    def __call__(self, t: float):
        '''Method to compute a value of the survival function S(t)'''
        # before minimal value
        if t < self.X[0]:
            return self.F[0]
        # after maximal value
        elif t >= self.X[-1]:
            if self.last_is_censored is False:
                return 0.0
            else:
                return self.F[-1]
        # between minimal and maximal values
        else:
            i = 0
            while True:
                if self.X[i] <= t < self.X[i+1]:
                    return self.F[i+1]
                i += 1

    def quantile(self, p: float):
        '''Method to compute the quantile S^-1(p)'''
        if p >= 1.0:
            return -sys.float_info.max
        elif p <= 0.0:
            return sys.float_info.max
        else:
            if p < self.F[-1]:
                if self.last_is_censored is True:
                    return sys.float_info.max
                else:
                    return self.X[-1]
            else:
                i = 0
                while True:
                    if self.F[i] > p >= self.F[i+1]:
                        return self.X[i]
                    i += 1

    def load(self, path_to_load):
        '''Load the fitted model.'''
        obj = pickle.load(open(path_to_load, 'rb'))
        self.X = obj.X
        self.F = obj.F
        self.last_is_censored = obj.last_is_censored

    def save(self, path_to_save):
        '''Saving the fitted model.'''
        s = pickle.dumps(self)
        fd = open(path_to_save, 'wb')
        fd.write(s)
        fd.close()

# entry point
if __name__ == '__main__':
    # initial data
    T = [16, 12, 11, 14]        # times
    C = [0, 0, 0, 1]            # events (failure - 0, right-censored - 1)

    # build the Kaplan-Meier estimator S_KM(t)
    kme = KaplanMeierEstimator(T, C)

    # predict S_KM(t=11.5)
    print(kme(11.5))
