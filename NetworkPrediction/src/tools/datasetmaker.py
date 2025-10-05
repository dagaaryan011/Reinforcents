import numpy as np
import pandas as pd
from datetime import date, timedelta
# from .noise import generate_price_samples
from tqdm import tqdm
# --- Corrected Stateless Indicators Class ---
class Indicators:
    @staticmethod
    def SMA_n(trend:list[float], n:int):
        if len(trend) < n:
            return None, None
        prices = np.array(trend[-n:])
        return np.mean(prices), np.std(prices, ddof=1)

    @staticmethod
    def EMA_n(trend:list[float], n:int):
        if len(trend) < n:
            return None
        price_series = pd.Series(trend)
        ema_series = price_series.ewm(span=n, adjust=False).mean()
        return ema_series.iloc[-1]

    @staticmethod
    def MACD(trend:list[float]):
        EMA_12 = Indicators.EMA_n(trend, 12)
        EMA_26 = Indicators.EMA_n(trend, 26)
        if EMA_12 is not None and EMA_26 is not None:
            return EMA_12 - EMA_26
        return None

    @staticmethod
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

    @staticmethod
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
    overbought = sum(1 for rsi in window if rsi > 70)
    oversold = sum(1 for rsi in window if rsi < 30)
    if overbought > oversold: return 1
    if oversold > overbought: return -1
    return 0
def get_rsi_conviction( rsi_history):
    rsi_short = get_rsi_signal(rsi_history,window_size=10)
    rsi_med = get_rsi_signal(rsi_history,window_size=30)
    rsi_long = get_rsi_signal(rsi_history,window_size=60)

    rsi_conviction = rsi_short*0.35 + rsi_med*0.4 + rsi_long*0.25
    if rsi_conviction > 0.9:
        return 1.5
    elif rsi_conviction < (-0.9):
        return -1.5
    else:
        return rsi_conviction    # rsi_short:float ,rsi_med: float,rsi_long:float
def get_stochastic_signal(k_history: list[float],d_history:list[float], window_size: int = 30):
    if len(k_history) < window_size:
        return 0
    window_k = k_history[-window_size:]
    window_d = d_history[-window_size:]
    oversold = 0
    overbought = 0
    for i in range(window_size):
        if window_k[i]> 80 and window_d[i]> 80 :
            overbought +=1
        elif window_k[i]<20 and window_d[i]<20 :    
            oversold +=1
    
    if overbought > oversold: return 1
    if oversold > overbought: return -1
    return 0
def get_stoch_conviction ( k_history:list[float],d_history:list[float]):
    k_short = k_history[-10:]
    d_short = d_history[-10:]
    k_med = k_history[-30:]
    d_med = d_history[-30:]
    k_long = k_history[-60:]
    d_long = d_history[-60:]
    stoch_short = get_stochastic_signal(k_short,d_short,window_size=10)
    stoch_med = get_stochastic_signal(k_med,d_med,window_size=30)
    stoch_long = get_stochastic_signal(k_long,d_long,window_size=60)

    stoch_conviction = 0.4*stoch_med + 0.3*stoch_long + 0.3*stoch_short
    if stoch_conviction > 0.9:
        return 1.5
    elif stoch_conviction < (-0.9):
        return -1.5
    else:
        return stoch_conviction 
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

    # This is the guard clause that fixes the error
    if smooth_posDM is None or smooth_negDM is None or avgTR is None or avgTR == 0:
        return None

    DIpos = 100 * (smooth_posDM / avgTR)
    DIneg = 100 * (smooth_negDM / avgTR)

    if (DIpos + DIneg) == 0:
        return 0.0

    dx = 100 * abs(DIpos - DIneg) / (DIpos + DIneg)
    return dx
def get_ADX(dx_history:list[float]):
    adx = Indicators.EMA_n(dx_history , n = 14)
    return adx


def generate_labels(price_path: list[float], look_ahead: int = 30, threshold: float = 0.005):
    labels = []
    for i in range(len(price_path) - look_ahead):
        if price_path[i + look_ahead] > price_path[i] * (1 + threshold): labels.append(1)
        elif price_path[i + look_ahead] < price_path[i] * (1 - threshold): labels.append(-1)
        else: labels.append(0)
    labels.extend([0] * look_ahead)
    return labels

# def create_training_dataset(price_path: list[float]):
    
#     print("Generating labels...")
#     labels = generate_labels(price_path)
    
#     print("Generating indicator signals and combining data...")
#     trend, macd_history, rsi_history, k_history,d_history,DMpos_history ,DMneg_history , true_range_history, dx_history = [], [], [], [],[],[],[],[],[]
#     training_data = []
#     min_warmup = 94  

#     for i, price in enumerate(tqdm((price_path),desc="Processing data")):
#         trend.append(price)
        
#         # update the raw indicator histories
#         macd_history.append(Indicators.MACD(trend))
#         rsi_history.append(Indicators.Smooth_RSI(trend))
#         k_history.append(Indicators.Stoch_Oscilator(trend))
#         DMpos, DMneg = get_DMs(trend)
#         DMneg_history.append(DMneg)
#         DMpos_history.append(DMpos)
       
#         if len(k_history) >= 3:
#             # Get the last 3 K values
#             last_three_k = k_history[-3:]

#             # Check if any of them are None before summing
#             if all(k is not None for k in last_three_k):
#                 d_value = sum(last_three_k) / 3
#                 d_history.append(d_value)
#             else:
#                 d_history.append(None) # Append None if data is not ready
            

#         if i >= min_warmup:
            
#             true_range_history.append(get_true_range(trend))
#             dx_history.append(get_DX(DMpos_history=DMpos_history,DMneg_history=DMneg_history,true_range_history=true_range_history))
#             macd_sig = macd_signal(macd_history)
#             rsi_sig = get_rsi_conviction(rsi_history)
#             stoch_sig =get_stoch_conviction(k_history,d_history)
#             status_sig = get_current_status(trend)
#             adx_sig = get_ADX(dx_history=dx_history)
#             #  input row
#             inputs = [
#                 macd_sig if macd_sig is not None else 0,
#                 rsi_sig if rsi_sig is not None else 0.0,
#                 stoch_sig if stoch_sig is not None else 0.0,
#                 status_sig if status_sig is not None else 0.0,
#                 adx_sig if adx_sig is not None else 0.0
#             ]
            
#             # 
#             output = labels[i]
            
#             training_data.append([inputs, output])
            
#     return np.array(training_data, dtype=object)
def create_training_dataset(price_path, sequence_length=30):
    print("Generating labels...")
    labels = generate_labels(price_path)

    print("Generating indicator signals...")
    trend, macd_history, rsi_history, k_history, d_history = [], [], [], [], []
    DMpos_history, DMneg_history, true_range_history, dx_history = [], [], [], []
    
    all_features = []
    min_warmup = 94

    for i, price in enumerate(tqdm(price_path, desc="Calculating All Features")):
        trend.append(price)
        
        macd_history.append(Indicators.MACD(trend))
        rsi_history.append(Indicators.Smooth_RSI(trend))
        k_history.append(Indicators.Stoch_Oscilator(trend))
        DMpos, DMneg = get_DMs(trend)
        DMneg_history.append(DMneg)
        DMpos_history.append(DMpos)
        
        if len(k_history) >= 3:
            last_three_k = k_history[-3:]
            if all(k is not None for k in last_three_k):
                d_value = sum(last_three_k) / 3
                d_history.append(d_value)
            else:
                d_history.append(None)
        else:
             d_history.append(None)

        if i >= min_warmup:
            true_range_history.append(get_true_range(trend))
            dx_history.append(get_DX(DMpos_history=DMpos_history, DMneg_history=DMneg_history, true_range_history=true_range_history))
            macd_sig = macd_signal(macd_history)
            rsi_sig = get_rsi_conviction(rsi_history)
            stoch_sig = get_stoch_conviction(k_history, d_history)
            status_sig = get_current_status(trend)
            adx_sig = get_ADX(dx_history=dx_history)
            
            inputs = [
                macd_sig if macd_sig is not None else 0,
                rsi_sig if rsi_sig is not None else 0.0,
                stoch_sig if stoch_sig is not None else 0.0,
                status_sig if status_sig is not None else 0.0,
                adx_sig if adx_sig is not None else 0.0
            ]
            all_features.append(inputs)
        else:
            all_features.append([0, 0.0, 0.0, 0.0, 0.0])
            true_range_history.append(None)
            dx_history.append(None)

    training_data = []
    print("Creating sequences...")
    for i in range(min_warmup + sequence_length, len(all_features)):
        sequence = all_features[i-sequence_length:i]
        output = labels[i]
        training_data.append([sequence, output])
        
    return np.array(training_data, dtype=object)

if __name__ == '__main__':
    end_date = date.today()
    start_date = end_date - timedelta(days=250)
    current_date = start_date
    
    full_price_path = []
    full_price_path = pd.read_csv("test_price_data.csv")['price'].tolist()
    total_days = (end_date - start_date).days

    # print(f"Fetching 5 years of data from {start_date} to {end_date}...")
    # with tqdm(total=total_days, desc="Fetching Data") as pbar:
    #     while current_date < end_date:
    #         daily_price_path    = generate_price_samples(current_date)
    #         while daily_price_path is None :
    #             current_date += timedelta(days=1)
    #             daily_price_path = generate_price_samples(current_date)
    #             pbar.update(1)
    #         full_price_path.extend(daily_price_path.tolist())    
    #         print(f"Fetched data for: {current_date}", end="\r")
    #         current_date += timedelta(days=1)
    #         pbar.update(1)
    
    # print("\nData fetching complete.")

    
        
    x = 15475
    for i in range(30):
        start_index = i * x
        end_index = (i + 1) * x
        price_subset = full_price_path[start_index:end_index]
        dataset_subset = create_training_dataset(price_subset)
        np.save(rf'D:\NetworkPrediction\training_data\training_data{i+1}.npy', dataset_subset)
    print("\n--- Dataset Generation Complete ---")
    
    