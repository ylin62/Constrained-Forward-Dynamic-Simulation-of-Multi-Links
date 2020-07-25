import numpy as np
import sympy as sp

class ExplictModel(object):
    '''
        build explict n chain bar model with uniform rod, no spring, 2-D
    '''
    def __init__(self, m, l, g=9.81, close_chain=True, external_input=None, energy_loss=None):
        super(ExplictModel, self).__init__()
                
        # physical parameters
        self.g = g
        self.close_chain = close_chain
        
        if not isinstance(m, np.ndarray):
            m = np.array(m)
        if not isinstance(l, np.ndarray):
            l = np.array(l)
            
        if close_chain:
            assert len(l) - len(m) == 1, 'closed chain should have one more rod length (ground rod) than rod mass'
            self.inertia = 1./12. * m * l[:-1]**2
        else:
            assert len(l) == len(m), 'open chain should have equal num of rod lenght and mass (no ground rod)'
            self.inertia = 1./12. * m * l**2
        self.m = m
        self.l = l
        self.n_rod = len(m)
        self.t = sp.symbols('t')
        
        # build generalized coordinates and velocities symbolic variables
        self.q, self.q_dot = [], []
        for i, mi in enumerate(m):
            xc, yc, theta = 'xc'+'_'+str(i+1), 'yc'+'_'+str(i+1), 'theta_'+str(i+1)        
            self.q.extend([sp.Function(xc)(self.t), sp.Function(yc)(self.t), sp.Function(theta)(self.t)])
        self.q = sp.Matrix(self.q)
        self.q_dot = self.q.diff(self.t)
        
    def lagrangian(self):
        
        T, V = 0, 0
        for i, m in enumerate(self.m):
            T += 0.5 * (m * (self.q_dot[i*self.n_rod + 0]**2 + self.q_dot[i*self.n_rod + 1]**2) + self.inertia[i] * self.q_dot[i*self.n_rod + 2]**2)
        
        for i, mi in enumerate(self.m):
            V += mi * self.g * self.q[i*self.n_rod + 1]
            
        Constrians = self.constrains()
        
        Lagrangian_multipliers = []
        for i in range(len(Constrians)):
            Lagrangian_multipliers.append(sp.symbols('lamda_'+str(i+1)))
        
        L = sp.Matrix([T - V]) + sp.Matrix(Lagrangian_multipliers).T * Constrians
        
        return L

    def constrains(self):
        
        Constrains = []
        for i in range(self.n_rod):
            expr = self.q[i*self.n_rod:i*self.n_rod+2]
            expr[0] -= self.l[i]/2 * sp.cos(self.q[i*self.n_rod + 2])
            expr[1] -= self.l[i]/2 * sp.sin(self.q[i*self.n_rod + 2])
            for j in range(i):
                theta = self.q[j*self.n_rod + 2]
                expr[0] -= self.l[j] * sp.cos(theta)
                expr[1] -= self.l[j] * sp.sin(theta)
            Constrains.extend(expr)
            
        if self.close_chain:
            expr = [-self.l[-1], 0]
            for i in range(len(self.m)):
                expr[0] += self.l[i] * sp.cos(self.q[i*self.n_rod + 2])
                expr[1] += self.l[i] * sp.sin(self.q[i*self.n_rod + 2])
            Constrains.extend(expr)
                    
        return sp.Matrix(Constrains)
        
        
if __name__ == "__main__":
    
    Demo = ExplictModel(m=[1, 1, 1], l=[1, 1, 1, 1], g=9.81, close_chain=True)
    L = Demo.lagrangian()
    t = Demo.t
    q = Demo.q
    q_dot = Demo.q_dot
    
    print(((L.jacobian(q_dot)).diff(t) - L.jacobian(q)).T)