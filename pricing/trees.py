import numpy as np


class TreePricer:
    def generate_tree(self, vol, intervals, dt):
        vol = np.asarray(vol)
        if vol.ndim == 0:
            vol = vol * np.array([1])
        intervals = np.asarray(intervals)
        if intervals.ndim == 0:
            intervals = intervals * np.array([1])
        dt = np.asarray(dt)
        no_steps = np.floor(intervals / dt)
        taus = intervals / no_steps
        up_moves = np.exp(vol * taus)
        down_moves = 1 / up_moves
        trees = [
            [
                up_moves[i] ** np.arange(j + 1)
                * down_moves[i] ** np.flip(np.arange(j + 1))
                for j in range(int(no_steps[i]) + 1)
            ]
            for i in range(intervals.size)
        ]
        if len(trees) == 1:
            trees = trees[0]
        return trees

    def compute_price_tree(self, prices, vol, intervals, dt):
        prices = np.asarray(prices)
        trees = self.generate_tree(vol, intervals, dt)
        if type(trees[0]) != list:
            price_trees = [prices * i for i in trees]
        else:
            price_trees = [[prices[i] * j for j in trees[i]] for i in range(len(trees))]
        return price_trees

    def compute_tree_expectation(self, payoffs, drift, vol, intervals):
        if type(payoffs[0]) != list:
            payoffs = [payoffs]
        payoffs = [np.asarray(payoff) for payoff in payoffs]
        drift = np.asarray(drift)
        if drift.ndim == 0:
            drift = drift * np.array([1])
        vol = np.asarray(vol)
        if vol.ndim == 0:
            vol = vol * np.array([1])
        intervals = np.asarray(intervals)
        if intervals.ndim == 0:
            intervals = intervals * np.array([1])
        no_steps = np.array([payoff.size - 1 for payoff in payoffs])
        dt = intervals / no_steps
        up_moves = np.exp(vol * dt)
        down_moves = 1 / up_moves
        p = (np.exp(drift * dt) - down_moves) / (up_moves - down_moves)
        expectations = np.full(len(payoffs), np.nan)
        for i in range(len(payoffs)):
            temp = payoffs[i]
            for j in range(no_steps[i]):
                temp = (temp[:-1] * p[i] + temp[1:] * (1 - p[i])) * np.exp(
                    -drift[i] * dt[i]
                )
            expectations[i] = temp[0]
        return expectations
