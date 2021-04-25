import numpy as np
from functools import partial
from scipy.stats import norm
from scipy import optimize

FWD = 'FWD'
CALL = 'CALL'
PUT = 'PUT'
STRADDLE = 'STRADDLE'

def option_price_interval(fwd, pv, strike, pt):
    """Calculates the European option no-arbitrage interval

    Args:
        fwd (float): Forward price
        pv (float): Discount factor
        strike (float): Strike
        pt (enum): CALL or PUT indicator

    Returns:
        (float, float): Lower and upper bound of interval
    """
    
    pt = pt.upper()
    if pt not in [CALL, PUT]:
        raise Exception('Unrecognized payoff type: ' + pt)

    if isinstance(fwd, (list, tuple, np.ndarray)):
        fwd_is_list = True
        fwd = np.asarray(fwd)
    else:
        fwd_is_list = False
        
    if isinstance(strike, (list, tuple, np.ndarray)):
        strike_is_list = True
        strike = np.asarray(strike)
    else:
        strike_is_list = False
    
    pv_fwd = pv * fwd
    pv_strike = pv * strike
    if pt == CALL:
        lb = np.maximum(pv_fwd - pv_strike, 0.0)
        if not fwd_is_list and strike_is_list:
            ub = np.full(strike.shape, pv_fwd)
        else:
            ub = pv_fwd
    else:
        lb = np.maximum(pv_strike - pv_fwd, 0.0)
        if not strike_is_list and fwd_is_list:
            ub = np.full(fwd.shape, pv_strike)
        else:
            ub = pv_strike
    
    return (lb,ub)

def option_intrinsic_value(spot, strike, pt):
    """Calculates the European option intrinsic ptice

    Args:
        spot (float): Spot price
        strike (float): Strike
        pt (enum): CALL or PUT indicator

    Returns:
        (float, float): Intrinsic option price
    """  
    
    pt = pt.upper()
    if pt not in [CALL, PUT]:
        raise Exception('Unrecognized payoff type: ' + pt)
    
    if isinstance(spot, (list, tuple, np.ndarray)):
        spot = np.asarray(spot)
        
    if isinstance(strike, (list, tuple, np.ndarray)):
        strike = np.asarray(strike)
    
    if pt == CALL:
        return np.maximum(spot - strike, 0.0)
    else:
        return np.maximum(strike - spot, 0.0)

def option_price(fwd, vol, pv, strike, texp, pt):
    """Calculates the Black Scholes price of a European option

    Args:
        fwd (float): Forward price
        vol (float): Volatility
        pv (float): Discount factor
        strike (float): Strike
        texp (float): Expiry time
        pt (enum): CALL or PUT indicator

    Returns:
    float: The Black Scholes option price
    """
    
    min_val = 0.0000000001
    vol = np.maximum(vol, min_val)
    texp = np.maximum(texp, min_val)
    
    pt = pt.upper()
    if pt not in [FWD, CALL, PUT, STRADDLE]:
        raise Exception('Unrecognized payoff type: ' + pt)
    
    if pt == FWD:
        return pv * (fwd - strike)
    
    sqrtvar = vol * np.sqrt(texp)
    d1 = (np.log(fwd/strike) + 0.5 * sqrtvar**2 ) / sqrtvar
    d2 = d1 - sqrtvar
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    
    c = pv * (fwd * nd1 - strike * nd2)
    if pt == CALL:
        return c
    else:
        p = c - pv * (fwd - strike)
        if pt == PUT:
            return c - pv * (fwd - strike)
        else:
            return c + p
      
def option_vol(price, fwd, pv, strike, texp, pt, min_vol = 0.0, max_vol = 20.0):
    """Implies the Black Scholes volatility for a given price for a European option

    Args:
        price (float): Option price
        fwd (float): Forward price
        pv (float): Discount factor
        strike (float): Strike
        texp (float): Expiry time
        pt (enum): CALL or PUT indicator
        min_vol (float): Lower bound of implied volatility
        max_vol (float): Upper bound of implied volatility

    Returns:
        float: The Black Scholes implied volatility
    """  
    
    bsvol = partial(option_price, fwd = fwd, pv = pv, strike = strike, texp = texp, pt = pt)
    def target(v, t):
        y = bsvol(vol = v) - t
        return y
    
    try: 
        vol = optimize.bisect(partial(target, t = price), min_vol, max_vol)
    except:
        vol = np.nan
    return vol
  
def option_risk(spot, vol, rate, strike, texp, pt):
    """Calculates the Black Scholes price and greeks of a European option

    Args:
        spot (float): Sopt price
        vol (float): Volatility
        rate (float): Interest rate
        strike (float): Strike
        texp (float): Expiry time
        pt (enum): CALL or PUT indicator

    Returns:
    dict: The Black Scholes option price and greeks in a dictionary with labels 'Price', 'Delta', 'Gamma', 'Vega'
    """
    
    min_val = 0.0000000001
    vol = np.maximum(vol, min_val)
    texp = np.maximum(texp, min_val)
    
    pt = pt.upper()
    if pt not in [CALL, PUT]:
        raise Exception('Unrecognized payoff type: ' + pt)

    pv = np.exp(-rate * texp)
    fwd = spot / pv
    
    sqrtvar = vol * np.sqrt(texp)
    d1 = (np.log(fwd/strike) + 0.5 * sqrtvar**2 ) / sqrtvar
    d2 = d1 - sqrtvar
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    
    c = pv * (fwd * Nd1 - strike * Nd2)
    if pt == CALL:
        price = c
        delta = Nd1
    else:
        price = c - pv * (fwd - strike)
        delta = Nd1 - 1.0

    nd1 = norm.pdf(d1)
    gamma = nd1 / (spot * sqrtvar)
    vega = spot * nd1 * np.sqrt(texp)
    
    results = {
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega
    }
    
    return results