from math import log, sqrt, exp
from scipy.stats import norm

class BlaScho:
    def __init__(self):
        self.Spot =  8400       # S = Current stock price (spot price)
        self.Strike = 8600        # K = Strike price (agreed future price in option contract)
        self.Time = 1       # T = Time to maturity (in years)
        self.Annual_return =  0.07       # r = Risk-free interest rate (annual, as a decimal),guaranteed amount of interest that can be earned without any risk of losing your money.
        self.Volatility =  0.18     # V = Volatility (standard deviation of stock returns)
        self.option = "call"

    def calculate_option_price(self):
        S=self.Spot
        K= self.Strike
        T = self.Time
        r = self.Annual_return
        v = self.Volatility
        option_type = self.option

        d1 = (log(S / K) + (r + 0.5 *( v** 2)) * T) / (v * sqrt(T))
        d2 = d1 - v * sqrt(T)

        premium_call = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        premium_put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        delta_call = norm.cdf(d1)
        delta_put = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * v * sqrt(T))

        theta_call = - (S * norm.pdf(d1) * v) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        theta_put = - (S * norm.pdf(d1) * v) / (2 * sqrt(T))  + r * K * exp(-r * T) * norm.cdf(-d2)

        vega = S * norm.pdf(d1) * sqrt(T)

        if option_type == 'call':
            return premium_call
        elif option_type == 'put':
            return premium_put
