import numpy as np
import sympy as sp
# np.seterr('raise')

class BaseModel(object):
    '''
        build explict n chain bar model with uniform rod, no spring, 2-D
    '''
    def __init__(self, m, l, close_chain=True):
        super(BaseModel, self).__init__()
                
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
        
        self.potiential_field = sp.Matrix(sp.symbols('g_1:%d'%(3*self.n_rod+1)))
        self.external_input = sp.Matrix(sp.symbols('f_1:%d'%(3*self.n_rod+1)))        
        self.damping = sp.Matrix(sp.symbols('c_1:%d'%(3*self.n_rod+1)))
        
        self.t = sp.symbols('t')
        
        self.generalized_coordinates()
        self.A = self.constrains.jacobian(self.q)
        self.A_dot = self.A.diff(self.t)
        
    def generalized_coordinates(self):
        
        # build generalized coordinates and velocities symbolic variables
        self.q, self.q_dot = [], []
        for i in range(self.n_rod):
            xc, yc, theta = 'x_c'+str(i+1), 'y_c'+str(i+1), 'theta_'+str(i+1)        
            self.q.extend([sp.Function(xc)(self.t), sp.Function(yc)(self.t), sp.Function(theta)(self.t)])
        self.q = sp.Matrix(self.q)
        self.q_dot = self.q.diff(self.t)
        
    @property
    def M(self):
        
        M = []
        for i in range(self.n_rod):
            M += [self.m[i], self.m[i], self.inertia[i]]
        
        M = np.diag(M)
        
        return sp.Matrix(M)
        
    @property
    def lagrangian(self):
                
        T = 0.5 * self.q_dot.T * self.M * self.q_dot
        V = (self.potiential_field.T * self.M) * self.q
        
        Lagrangian_multipliers = []
        for i in range(len(self.constrains)):
            Lagrangian_multipliers.append(sp.symbols('lamda_'+str(i+1)))
        
        L = sp.Matrix([T - V]) + sp.Matrix(Lagrangian_multipliers).T * self.constrains
        
        return L

    @property
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
        pass
    
    def sim(self):
        pass

class ExplictModel(BaseModel):
    '''
        explicit calculate lagrangian multipliers
    '''
    def __init__(self, m, l, close_chain=True):
        super().__init__(m, l, close_chain=close_chain)
        self.system_gov()
    
    def system_gov(self):
        
        f = self.external_input + self.M * self.potiential_field
        a = np.block([[self.M, self.A.T], 
                      [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))]])
        b = np.concatenate([f, -self.A_dot * self.q_dot])
        
        self.a = sp.lambdify(self.q, sp.Matrix(a))
        self.b = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), sp.Matrix(b))
    
    def sim(self, t, y, f=None, g=None, c=None):
        
        if f is not None:
            assert len(f) == 3*self.n_rod, 'input needs to be a vector of length 3*n \
                corresponding to [x, y, torque]'
            f = np.array(f)
        else:
            f = np.zeros(3*self.n_rod)
        if g is not None:
            assert len(g) == 3*self.n_rod, 'potiential energy field n*[x, y, tau...]'
            g = np.array(g)
        else:
            g = np.zeros(3*self.n_rod)
        if c is not None:
            assert len(c) == 3*self.n_rod
            c = np.array(c)
            print("in development")
        else:
            c = np.zeros(3*self.n_rod)

        a = self.a(*y[:3*self.n_rod])
        input_f = np.append(y, [f, g])
        b = self.b(*input_f)
        c = np.linalg.solve(a, b)
                
        return np.append(y[3*self.n_rod:], c[:len(self.q)])
    
class ApproximateModel(BaseModel):
    """Approximate constrain forces as spring, violating of constrain resulting constrain force"""
    def __init__(self, m, l, k, close_chain=True):
        super(ApproximateModel, self).__init__(m, l, close_chain=close_chain)
        assert len(k) == len(self.constrains), 'constrain stiffness should equal to num of constrains'
        self.k = np.diag(k)
        self.system_gov()
        
    def system_gov(self):
        
        f = self.external_input + self.M * self.potiential_field
        b = self.M.inv() * (f - self.A.T * sp.Matrix(self.k) * self.constrains)
        self.b = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), sp.Matrix(b))
        
    def sim(self, t, y, f=None, g=None, c=None):
        
        input_f = np.append(y, [f, g])
        b = self.b(*input_f)
        
        return np.append(y[3*self.n_rod:], b)
    
class ProjectModel(BaseModel):
    """docstring for NullSpaceModel"""
    def __init__(self, m, l, close_chain=True):
        super(ProjectModel, self).__init__(m, l, close_chain=close_chain)
        self.S = self.A.nullspace()[-1]
        self.v = self.q[-1]
        self.system_gov()
        
    def system_gov(self):
        
        f = self.external_input + self.M * self.potiential_field
        
        odefunc = sp.Matrix([[self.S * self.v], 
                             [(self.S.T * self.M * self.S).inv() * 
                              (self.S.T * f - self.S.T * self.M * self.S.diff(self.t) * self.v)]])
        
        self.odefunc = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), odefunc)
        self.S_func = sp.lambdify(self.q, self.S)
        
    def sim(self, t, y, f=None, g=None, c=None):
        
        q_dot = self.S_func(*y[:-1]) * y[-1]
        f1 = np.append(y[:-1], q_dot)
        input_f = np.append(f1, [f, g])
        ode = self.odefunc(*input_f)
        
        return ode[:, 0]
        
        
if __name__ == "__main__":
    
    import time
    from scipy.integrate import solve_ivp,  odeint
    import matplotlib.pyplot as plt
    
    m = [1, 1, 1]
    l = [1, 4, 2.5, 3]
    y = np.append([0, 0.5, np.pi/2, 1.8765, 1.692, 0.3533, 3.3765, 1.192, -1.8767], np.zeros(9))
    g = [0, -9.81, 0, 0, -9.81, 0, 0, -9.81, 0]
    f = [0, 0, 5, 0, 0, 0, 0, 0, 0]
    
    #####################################Test Explict Model##########################################
    t0 = time.time()
    Demo = ExplictModel(m=m, l=l, close_chain=True)
    # t0 = time.time()
    # Demo.sim(t=0, y=y, f=f, g=g, c=None)
    # print(time.time() - t0)
    # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    t1 = time.time()
    sol = solve_ivp(Demo.sim, [0, 10], y, method='DOP853', args=(f, g, None))
    print(time.time() - t1)
    plt.figure()
    plt.plot(sol.t, sol.y[2])
    plt.figure()
    plt.plot(sol.t, sol.y[5])
    plt.figure()
    plt.plot(sol.t, sol.y[8])
    plt.show()
    
    print(Demo.lagrangian)
    
    ####################################Test Approximate Model######################################
    # k = np.tile([10e6], 8)
    # Demo = ApproximateModel(m=m, l=l, k=k, close_chain=True)
    # # t0 = time.time()
    # # Demo.sim(t=0, y=y, f=f, g=g, c=None)
    # # print(time.time() - t0)
    # # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    # t1 = time.time()
    # sol = solve_ivp(Demo.sim, [0, 10], y, method='RK23', args=(f, g, None))
    # print(sol.t.shape)
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()
    
    ####################################Test Projection Model########################################
    y = np.append([0, 0.5, np.pi/2, 1.8765, 1.692, 0.3533, 3.3765, 1.192, -1.8767], 0)
    # Demo = ProjectModel(m, l, close_chain=True)
    # # t0 = time.time()
    # # Demo.sim(t=0, y=y, f=f, g=g, c=None)
    # # print(time.time() - t0)
    # # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    # t1 = time.time()
    # sol = solve_ivp(Demo.sim, [0, 0.5], y, method='DOP853', args=(f, g, None))
    # print(sol.t.shape)
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()
    
    ####################################Tret odeint##################################################
    # t = np.linspace(0, 1, 2000)
    # Demo = ProjectModel(m, l, close_chain=True)
    # t1 = time.time()
    # sol = odeint(Demo.sim, t=t, y0=y, args=(f, g, None), tfirst=True)
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(t, sol[:, 2])
    # plt.figure()
    # plt.plot(t, sol[:, 5])
    # plt.figure()
    # plt.plot(t, sol[:, 8])
    # plt.show()