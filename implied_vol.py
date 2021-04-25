import numpy as np
import black_scholes as bs

def build_tanh_prop_moneyness(w, PV):
    """TANH proportional moneyness factory

    Args:
        w (float): Cutoff velocity parameter
        PV (float): Discount Factor

    Returns:
        moneyness: TANH proportional moneyness functor
    """   
    def moneyness(K, KRef):
        """TANH proportional moneyness

        Args:
            K (float): Strike
            KRef (float): Strike Reference

        Returns:
            float: TANH proportional Moneyness
        """           
        
        fwd_ref = KRef / PV
        pm = (K - fwd_ref) / fwd_ref
        m = w * np.tanh(pm / w)
        return m
    
    return moneyness
    
def build_quad_prop_moneyness(PV):
    """Quadratic proportional moneyness factory

    Args:
        PV (float): Discount Factor

    Returns:
        moneyness: quadratic proportional moneyness functor
    """   
    
    def moneyness(K, KRef):
        """Quadratic proportional moneyness

        Args:
            K (float): Strike
            KRef (float): Strike Reference

        Returns:
            float: Quadratic proportional Moneyness
        """                  
        fwd_ref = KRef / PV
        pm = (K - fwd_ref) / fwd_ref
        m = pm
        return m
        
    return moneyness
  
class IV_Quad_F:
    """Implied Volatility model as a quadratic of generalised moneyness
    """                  
    
    def __init__(self, KRef, a, b, c, f):
        """Constructor

        Args:
            KRef (float): Strike Reference
            a (float): ATM Vol
            b (float): Slope around ATM f'(0)
            c (float): Convexity around ATM f''(0)
            f (functor): Moneyness functor
        """                  
        
        self.KRef = KRef
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        
        return
    
    def get_strike_ref(self):
        """Accessor to KRef
        
        Returns:
            float: Strike Reference

        """               
        return self.KRef
    
    def implied_vol(self, K, T):
        """Implied Volatility for requested strike and maturity

        Args:
            K (float): Strike
            T (float): Expiry

        Returns:
            float: The calculated implied volatility
        """            
        
        m = self.f(K, self.KRef)
        y = self.a + self.b * m + 0.5 * self.c * m**2
        return y
      
def implied_distribution(strikes, T, spot, PV, IVCalc):
    """Calculates implied quantities from the volatility smile for requested strikes on a given maturity

    Args:
        strikes ([float]): Strikes
        T (float): Expiry
        spot (float): Spot price
        PV (float): Discount factor
        IVCalc (object): Implied volatility calculator

    Returns:
        ([float], [float], [float], [float]): The calculated implied volatility, option price, implied distribution, implied density
    """           
    
    epsilon = 0.0001 * IVCalc.get_strike_ref()
    fwd = spot / PV
    
    strikes_up = strikes * (1.0 + epsilon)
    vols_up = IVCalc.implied_vol(strikes_up, T)
    v_up = bs.option_price(fwd, vols_up, 1.0, strikes_up, T, bs.PUT)
    
    vols = IVCalc.implied_vol(strikes, T)
    v = bs.option_price(fwd, vols, 1.0, strikes, T, bs.PUT)

    strikes_dn = strikes * (1.0 - epsilon)
    vols_dn = IVCalc.implied_vol(strikes_dn, T)
    v_dn = bs.option_price(fwd, vols_dn, 1.0, strikes_dn, T, bs.PUT)
    
    dist = (v_up - v_dn) / (strikes_up - strikes_dn)
    dens = (v_up - 2.0 * v + v_dn) / ((strikes_up - strikes)*(strikes - strikes_dn))
    
    return vols, v, dist, dens