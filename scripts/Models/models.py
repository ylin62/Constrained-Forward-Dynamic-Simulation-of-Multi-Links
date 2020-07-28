import numpy as np
import sympy as sp
import time
# np.seterr('raise')

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
        self.system_gov()
        
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
        for i in range(len(self.Constrains)):
            Lagrangian_multipliers.append(sp.symbols('lamda_'+str(i+1)))
        
        L = sp.Matrix([T - V]) + sp.Matrix(Lagrangian_multipliers).T * self.Constrains
        
        return L

    def constrains(self):
        
        Constrains = []
        for i in range(self.n_rod):
            expr = self.q[i*3:i*3+2]
            expr[0] -= self.l[i]/2 * sp.cos(self.q[i*3 + 2])
            expr[1] -= self.l[i]/2 * sp.sin(self.q[i*3 + 2])
            for j in range(i):
                theta = self.q[j*3 + 2]
                expr[0] -= self.l[j] * sp.cos(theta)
                expr[1] -= self.l[j] * sp.sin(theta)
            Constrains.extend(expr)
            
        if self.close_chain:
            expr = [-self.l[-1], 0]
            for i in range(len(self.m)):
                expr[0] += self.l[i] * sp.cos(self.q[i*3 + 2])
                expr[1] += self.l[i] * sp.sin(self.q[i*3 + 2])
            Constrains.extend(expr)
        
        Constrains = sp.Matrix(Constrains)
        
        return Constrains
    
    def system_gov(self):
        
        f = self.external_input + self.M.diagonal() * self.potiential_field
        a = np.block([[self.M, self.A.T], 
                      [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))]])
        b = np.concatenate([f.reshape(-1, 1), -self.A_dot * self.q_dot])
        
        self.a = sp.lambdify(self.q, sp.Matrix(a))
        self.b = sp.lambdify(sp.Matrix([self.q, self.q_dot]), sp.Matrix(b))
    
    def sim(self, t, y):

        a = self.a(*y[:3*self.n_rod])
        b = self.b(*y)
        c = np.linalg.solve(a, b)
                
        return np.append(y[3*self.n_rod:], c[:len(self.q)])
        
if __name__ == "__main__":
    
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    
    Demo = ExplictModel(m=[1, 1, 1], l=[1, 4, 2.5, 3], 
                        potiential_field=[0, -9.81, 0, 0, -9.81, 0, 0, -9.81, 0], 
                        close_chain=True, 
                        external_input=[0, 0, 5, 0, 0, 0, 0, 0, 0], 
                        damping=None)
    
    y = np.append([0, 0.5, np.pi/2, 1.8765, 1.692, 0.3533, 3.3765, 1.192, -1.8767], np.zeros(9))
    
    # print(Demo.sim(t=0, y=y))
    
    # sol = []
    # for t in np.linspace(0, 10, 1000):
    #     y_dot = Demo.sim(t=t, y=y)
    #     y += y_dot * 0.01
    #     sol += [y]
    # sol = np.array(sol)
    # t = np.linspace(0, 10, 1000)
    # plt.figure()
    # plt.plot(t, sol.T[2])
    # plt.figure()
    # plt.plot(t, sol.T[5])
    # plt.figure()
    # plt.plot(t, sol.T[8])
    # plt.show()
    
    sol = solve_ivp(Demo.sim, [0, 10], y, method='RK45')
    
    plt.figure()
    plt.plot(sol.t, sol.y[2])
    plt.figure()
    plt.plot(sol.t, sol.y[5])
    plt.figure()
    plt.plot(sol.t, sol.y[8])
    plt.show()
    
    # l1 = [[-0.5, 0.5], [0, 0]]
    # l2 = [[-2, 2], [0, 0]]
    # l3 = [[-1.25, 1.25], [0, 0]]
    
    # def rot(h):
    #     return np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])

    # bar1 = np.dot(rot(np.pi/2), l1) + np.array([0, 0.5]).reshape(-1, 1)
    # bar2 = np.dot(rot(0.3533), l2) + np.array([1.8765, 1.692]).reshape(-1, 1)
    # bar3 = np.dot(rot(-1.8767), l3) + np.array([3.3765, 1.192]).reshape(-1, 1)

    # plt.figure()
    # plt.plot(bar1[0], bar1[1])
    # plt.plot(bar2[0], bar2[1])
    # plt.plot(bar3[0], bar3[1])
    # plt.show()