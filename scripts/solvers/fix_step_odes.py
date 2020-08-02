from collections import namedtuple
import numpy as np

Solution = namedtuple('Solution', ['t', 'y'])

def ode1(func, tspan, y0, args=None):
    
    if args is not None:
        func = lambda t, y, func=func: func(t, y, *args)
    
    h = np.diff(tspan)
    sol = Solution
    sol.t = tspan
    sol.y = [y0]
    y = y0
    for i, t in enumerate(tspan[1:]):
        y += h[i] * func(t, y)
        sol.y = np.vstack([sol.y, y])
    sol.y = sol.y.T
    
    return sol

def ode2(func, tspan, y0, **kwargs):
    
    if args is not None:
        func = lambda t, y, func=func: func(t, y, *args)
        
    h = np.diff(tspan)
    sol = Solution
    sol.t = tspan
    sol.y = [y0]
    y = y0
    for i, t in enumerate(tspan[1:]):
        k1 = func(t, y)
        k2 = func(t+h[i], y+h[i]*k1)
        y += h[i]/2 * (k1 + k2)
        sol.y = np.vstack([sol.y, y])
    sol.y = sol.y.T
    
    return sol

def ode3(func, tspan, y0, **kwargs):
    
    if args is not None:
        func = lambda t, y, func=func: func(t, y, *args)
    
    h = np.diff(tspan)
    sol = Solution
    sol.t = tspan
    sol.y = [y0]
    y = y0
    for i, t in enumerate(tspan[1:]):
        k1 = func(t, y)
        k2 = func(t+0.5*h[i], y+0.5*h[i]*k1)
        k3 = func(t+0.75*h[i], y+0.75*h[i]*k2)
        y += h[i]/9 * (2*k1 + 3*k2 + 4*k3)
        sol.y = np.vstack([sol.y, y])
    sol.y = sol.y.T
    
    return sol

def ode4(func, tspan, y0, args=None):
    
    if args is not None:
        func = lambda t, y, func=func: func(t, y, *args)
    
    h = np.diff(tspan)
    sol = Solution
    sol.t = tspan
    sol.y = y0
    y = y0
    for i, t in enumerate(tspan[1:]):
        k1 = func(t, y)
        k2 = func(t+0.5*h[i], y+0.5*h[i]*k1)
        k3 = func(t+0.5*h[i], y+0.5*h[i]*k2)
        k4 = func(t+h[i], y+h[i]*k3)
        y += h[i]/6 * (k1 + 2*k2 + 2*k3 + k4)
        sol.y = np.vstack([sol.y, y])
    sol.y = sol.y.T
    
    return sol

def ode5(func, tspan, y0, args=None):
    pass

if __name__ == "__main__":
    pass