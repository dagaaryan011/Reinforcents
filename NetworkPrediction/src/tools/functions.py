# In functions.py

import numpy as np
import pandas as pd

class Indicators:
    
    def SMA_n(trend:list[float], n:int):
        if len(trend) < n:
            return None, None
        prices = np.array(trend[-n:])
        return np.mean(prices), np.std(prices, ddof=1)

    
    def EMA_n(trend:list[float], n:int):
        if len(trend) < n:
            return None
        price_series = pd.Series(trend)
        ema_series = price_series.ewm(span=n, adjust=False).mean()
        return ema_series.iloc[-1]

    
    def MACD(trend:list[float]):
        EMA_12 = Indicators.EMA_n(trend, 12)
        EMA_26 = Indicators.EMA_n(trend, 26)
        if EMA_12 is not None and EMA_26 is not None:
            return EMA_12 - EMA_26
        return None

    
    def Smooth_RSI(trend:list[float], n:int=14):
        if len(trend) < n + 1:
            return None
        price_series = pd.Series(trend)
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=n - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=n - 1, adjust=False).mean()
        last_avg_loss = avg_loss.iloc[-1]
        if last_avg_loss == 0:
            return 100.0
        rs = avg_gain.iloc[-1] / last_avg_loss
        return 100 - (100 / (1 + rs))

    
    def Stoch_Oscilator(trend:list[float], n:int=14):
        if len(trend) < n:
            return None
        window = np.array(trend[-n:])
        low = np.min(window)
        high = np.max(window)
        if high == low:
            return 50.0
        closing_price = trend[-1]
        return 100 * (closing_price - low) / (high - low)


def macd_signal(macd_history: list[float], window_size: int = 30):
    if len(macd_history) < (2 * window_size) + 9:
        return 0
    end_window = macd_history[-window_size:]
    start_window = macd_history[-(2 * window_size):-window_size]
    macd_end = end_window[-1]
    macd_start = start_window[-1]
    signal_end = Indicators.EMA_n(end_window, n=9)
    signal_start = Indicators.EMA_n(start_window, n=9)
    if signal_end is None or signal_start is None:
        return 0
    if (macd_start - signal_start) <= 0 and (macd_end - signal_end) > 0:
        return 1
    elif (macd_start - signal_start) >= 0 and (macd_end - signal_end) < 0:
        return -1
    else:
        return 0

def get_rsi_signal(rsi_history: list[float], window_size: int = 30):
    if len(rsi_history) < window_size:
        return 0
    window = rsi_history[-window_size:]
    overbought = sum(1 for rsi in window if rsi and rsi > 70)
    oversold = sum(1 for rsi in window if rsi and rsi < 30)
    if overbought > oversold: return 1
    if oversold > overbought: return -1
    return 0

def get_rsi_conviction(rsi_history,coeffs:list,window_size:tuple = (10,30,60)):
    rsi_short = get_rsi_signal(rsi_history,window_size=window_size[0])
    rsi_med = get_rsi_signal(rsi_history,window_size=window_size[1])
    rsi_long = get_rsi_signal(rsi_history,window_size=window_size[2])
    rsi_conviction = rsi_short*coeffs[0] + rsi_med*coeffs[1] + rsi_long*coeffs[2]
    if rsi_conviction > 0.9: return 1.5
    elif rsi_conviction < (-0.9): return -1.5
    else: return rsi_conviction
    
def get_stochastic_signal(k_history: list[float],d_history:list[float], window_size: int = 30):
    if len(k_history) < window_size or len(d_history) < window_size:
        return 0
    window_k = [k for k in k_history[-window_size:] if k is not None]
    window_d = [d for d in d_history[-window_size:] if d is not None]
    oversold = 0
    overbought = 0
    for i in range(min(len(window_k), len(window_d))):
        if window_k[i] > 80 and window_d[i] > 80 :
            overbought +=1
        elif window_k[i] < 20 and window_d[i] < 20 :    
            oversold +=1
    if overbought > oversold: return 1
    if oversold > overbought: return -1
    return 0
    
def get_stoch_conviction (k_history:list[float],d_history:list[float],coeffs:list,window_size:tuple = (10,30,60)):
    stoch_short = get_stochastic_signal(k_history, d_history, window_size=window_size[0])
    stoch_med = get_stochastic_signal(k_history, d_history, window_size=window_size[1])
    stoch_long = get_stochastic_signal(k_history, d_history, window_size=window_size[2])
    stoch_conviction = coeffs[0]*stoch_med + coeffs[1]*stoch_long + coeffs[2]*stoch_short
    if stoch_conviction > 0.9: return 1.5
    elif stoch_conviction < (-0.9): return -1.5
    else: return stoch_conviction

def get_current_status(trend: list[float]):
    if not trend: return 0.0
    day_trend = trend[-(len(trend) % 375):] if len(trend) % 375 != 0 else trend[-375:]
    if not day_trend: return 0.0
    high, low, current = np.max(day_trend), np.min(day_trend), day_trend[-1]
    if high == low: return 0.0
    midpoint = (high + low) / 2
    return (current - midpoint) / (high - midpoint) if high != midpoint else 0.0

def get_DMs (trend:list[float]):
   if len(trend)< 2:
       return None , None
   DMpos = 0
   DMneg = 0
   trend_window = trend[-14:]
   prev_high = max(trend_window[:-1])
   prev_low = min(trend_window[:-1])
   curr_high = max (trend_window)
   curr_low  = min(trend_window)
   upmove = curr_high - prev_high
   downmove = prev_low - curr_low
   if upmove > downmove and upmove > 0 :
       DMpos = upmove
   if downmove > upmove and downmove > 0 :
       DMneg= downmove
   return DMpos , DMneg

def get_true_range(trend:list[float]):
    if len(trend) < 15:
        return None
    curr_high = max(trend[-14:])
    curr_low = min(trend[-14:])
    prev_close = trend[-14]
    
    true_range = max(curr_high-curr_low,abs(curr_high-prev_close),abs(curr_low - prev_close))
    return true_range

def get_DX(DMpos_history:list[float] , DMneg_history:list[float], true_range_history:list[float]):
    smooth_posDM = Indicators.EMA_n(DMpos_history, n=14)
    smooth_negDM = Indicators.EMA_n(DMneg_history, n=14)
    avgTR = Indicators.EMA_n(true_range_history, n=14)
    if smooth_posDM is None or smooth_negDM is None or avgTR is None or avgTR == 0:
        return 0.0
    DIpos = 100 * (smooth_posDM / avgTR)
    DIneg = 100 * (smooth_negDM / avgTR)
    if (DIpos + DIneg) == 0:
        return 0.0
    dx = 100 * abs(DIpos - DIneg) / (DIpos + DIneg)
    return dx

def get_ADX(dx_history:list[float]):
    if not dx_history or dx_history[-1] is None:
        return None
    adx = Indicators.EMA_n(dx_history , n = 14)
    return adx
def calculate_historical_volatility(price_history: list, window: int = 30) -> float:
    """
    Calculates the annualized historical volatility over a given window.
    
    Returns:
        float: Annualized volatility, or a default value if not enough data.
    """
    # Use a default volatility if history is too short
    if len(price_history) < window + 1:
        return 0.20 # Default 20% volatility

    # Get the most recent 'window' of prices
    prices = np.array(price_history[-window:])
    
    # Calculate daily log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Calculate the standard deviation of log returns
    daily_std_dev = np.std(log_returns)
    
    # Annualize the volatility (assuming daily data for simplicity)
    # If your steps are minutes, this number would be much larger.
    # e.g., sqrt(252 * 375) for 375 minutes in a trading day.
    annualized_volatility = daily_std_dev * np.sqrt(252) 
    
    return annualized_volatility