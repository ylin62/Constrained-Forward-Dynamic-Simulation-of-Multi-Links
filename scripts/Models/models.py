import numpy as np
import sympy as sp
import time

class ExplictModel(object):
    '''
        build explict n chain bar model with uniform rod, no spring, 2-D
    '''
    def __init__(self, m, l, potiential_field=None, 
                 close_chain=True, external_input=None, damping=None):
        super(ExplictModel, self).__init__()
                
        # physical parameters

        self.close_chain = close_chain
        
        m = np.array(m)
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
        self.M = self.inertia_mtx()
        
        self.potiential_field = np.zeros(3*self.n_rod)
        if potiential_field is not None:
            assert len(potiential_field) == 3*self.n_rod, 'potiential energy field n*[x, y, tau...]'
            self.potiential_field = np.array(potiential_field)            
        
        self.external_input = np.zeros(3*self.n_rod)    
        if external_input is not None:
            assert len(external_input) == 3*self.n_rod, 'input needs to be a vector of length 3*n \
                corresponding to [x, y, torque]'
            self.external_input = np.array(external_input)
        
        self.damping = np.zeros(3*self.n_rod)
        if damping is not None:
            assert len(damping) == self.n_rod, 'only give dampings to joints, not including ground rod'
            self.damping[2::3] = damping
        
        self.t = sp.symbols('t')
        
        # build generalized coordinates and velocities symbolic variables
        self.q, self.q_dot = [], []
        for i, mi in enumerate(m):
            xc, yc, theta = 'x_c'+str(i+1), 'y_c'+str(i+1), 'theta_'+str(i+1)        
            self.q.extend([sp.Function(xc)(self.t), sp.Function(yc)(self.t), sp.Function(theta)(self.t)])
        self.q = sp.Matrix(self.q)
        self.q_dot = self.q.diff(self.t)
        
        self.Constrains = self.constrains()
        self.A = self.Constrains.jacobian(self.q)
        self.A_dot = self.A.diff(self.t)
        
    def inertia_mtx(self):
        
        M = []
        for i in range(self.n_rod):
            M += [self.m[i], self.m[i], self.inertia[i]]
        
        M = np.diag(M)
        
        return M
        
    def lagrangian(self):
                
        T = 0.5 * self.q_dot.T * self.M * self.q_dot
        V = (self.potiential_field @ self.M).reshape(1, -1) * self.q
        
        Lagrangian_multipliers = []
        for i in range(len(self.Constrians)):
            Lagrangian_multipliers.append(sp.symbols('lamda_'+str(i+1)))
        
        L = sp.Matrix([T - V]) + sp.Matrix(Lagrangian_multipliers).T * self.Constrians
        
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
        
        Constrains = sp.Matrix(Constrains)
        
        return Constrains
    
    def system_gov(self, t, y):
        
        t0 = time.time()
        
        y_q, y_q_dot = np.split(y, 2)
        
        sub_q = list(map(lambda x, y: (x, y), self.q, y_q))
        sub_q_dot = list(map(lambda x, y: (x, y), self.q_dot, y_q_dot))
        
        f = self.external_input + self.M.diagonal() * self.potiential_field
        
        a = np.block([[self.M, self.A.T], 
                      [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))]])

        t_sub = time.time()
        b = np.concatenate([f.reshape(-1, 1), -self.A_dot * self.q_dot])
        print(time.time() - t_sub)
        
        t1 = time.time()
        print(t1 - t0)
        
        a = sp.Matrix(a).subs(sub_q)
        b = sp.Matrix(b).subs(sub_q_dot)
        b = b.subs(sub_q)
        
        t2 = time.time()
        print(t2 - t1)
        
        c = np.linalg.solve(np.array(a).astype(float), np.array(b).astype(float))
        
        print(time.time() - t2)
        
        return np.append(y_q_dot, c[0:len(self.q)])
        
if __name__ == "__main__":
    
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    
    Demo = ExplictModel(m=[1, 1, 1], l=[1, 4, 2.5, 3], 
                        potiential_field=[0, 9.81, 0, 0, 9.81, 0, 0, 9.81, 0], 
                        close_chain=True, 
                        external_input=[0, 0, 5, 0, 0, 0, 0, 0, 0], 
                        damping=None)
    
    y = np.append([0, 0.5, np.pi/2, 1.876, 1.692, 0.353, 3.376, 1.192, 1.265], np.zeros(9))
    
    test = Demo.system_gov(t=0, y=y)
    
    print(test)
    
    # sol = solve_ivp(Demo.system_gov, [0, 1], y, method='RK23')
    
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()
