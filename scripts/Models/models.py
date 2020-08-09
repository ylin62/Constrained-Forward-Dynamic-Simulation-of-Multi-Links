import numpy as np
import sympy as sp
from scipy.optimize import fsolve
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
    
    def initial_condition(self, y0, y_dot):
        
        y0 = np.array(y0)
        valid_input = np.where(y0 != None)[0]
        if valid_input.size < len(self.q):
            y0_left = np.delete(np.arange(len(self.q)), valid_input)
            c = self.constrains
            for i in valid_input:
                c = c.subs(self.q[i], y0[i])
            q = np.delete(self.q, valid_input)
            func = sp.lambdify(q, c)
            root = fsolve(lambda x, func=func: func(*x).flatten(), np.zeros(len(self.q) - len(valid_input)))
            pos = np.zeros(len(self.q))
            pos[valid_input] = y0[valid_input]
            pos[y0_left] = root
        else:
            pos = y0
        
        y_dot = np.array(y_dot)
        valid_dot = np.where(y_dot != None)[0]
        if valid_dot.size < len(self.q_dot):
            vel_constrains = self.constrains.diff(self.t)
            func_dot = sp.lambdify(sp.Matrix([self.q, self.q_dot]), vel_constrains)
            
            input_root = np.zeros(2*len(self.q))
            input_root[:len(self.q)] = pos
            input_root[valid_dot + len(self.q)] = y_dot[valid_dot]

            idx_left = np.delete(np.arange(len(self.q)) + len(self.q), valid_dot)

            def func(x, fun):
                input_root[idx_left] = x
                return fun(*input_root).flatten()
            
            root_dot = fsolve(func, np.zeros(len(self.q) - len(valid_dot)), args=func_dot)
            vel = np.zeros(len(self.q))
            vel[valid_dot] = y_dot[valid_dot]
            vel[np.delete(np.arange(len(self.q)), valid_dot)] = root_dot
        else:
            vel = y_dot

        return pos, vel
    
    def system_gov(self):
        pass
    
    def sim(self):
        pass
    
    def get_multipliers(self):
        
        f = self.external_input + self.M * self.potiential_field
        a = np.block([[self.M, self.A.T], 
                      [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))]])
        b = np.concatenate([f, -self.A_dot * self.q_dot])
        
        a = sp.lambdify(self.q, sp.Matrix(a))
        b = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), sp.Matrix(b))
        
        return a, b

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
        b = self.M.inv() * (f - self.A.T * (sp.Matrix(self.k) * self.constrains))
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
        
        input_f = np.append(y, [f, g])
        b = self.b(*input_f)
        
        return np.append(y[3*self.n_rod:], b)
    
class ProjectModel(BaseModel):
    """docstring for NullSpaceModel"""
    def __init__(self, m, l, close_chain=True):
        super(ProjectModel, self).__init__(m, l, close_chain=close_chain)
        free_q = len(self.q) - len(self.constrains)
        self.S = sp.Matrix(self.A.nullspace()).reshape(free_q, len(self.q)).T
        self.v = sp.Matrix(self.q_dot[-free_q:])
        self.system_gov()
        
    def system_gov(self):
        
        f = self.external_input + self.M * self.potiential_field

        self.a = sp.lambdify(self.q, self.S.T * self.M * self.S)
        self.b = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), 
                             self.S.T * f - self.S.T * self.M * self.S.diff(self.t) * self.v)

        # odefunc = sp.Matrix([[self.S * self.v], 
        #                      [(self.S.T * self.M * self.S).inv() * 
        #                       (self.S.T * f - self.S.T * self.M * self.S.diff(self.t) * self.v)]])
        # self.odefunc = sp.lambdify(sp.Matrix([self.q, self.q_dot, self.external_input, self.potiential_field]), odefunc)
        
        self.S_func = sp.lambdify(self.q, self.S)
        
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
        
        q_dot = self.S_func(*y[:len(self.q)]) @ y[len(self.q):]
        f1 = np.append(y[:len(self.q)], q_dot)
        input_f = np.append(f1, [f, g])
        
        a = self.a(*y[:len(self.q)])
        b = self.b(*input_f)
        ode = np.append(q_dot, np.linalg.solve(a, b))
        
        # ode = self.odefunc(*input_f)
        # return ode[:, 0]
        
        return ode
    
class ExplictAlt(ExplictModel):
    """With seperate part constrains, join at second to last joint"""
        
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
            expr = self.q[(self.n_rod-1)*3:(self.n_rod-1)*3+2]
            expr[0] += -self.l[-1] - self.l[-2]/2 * sp.cos(self.q[3*(self.n_rod-1) + 2])
            expr[1] += - self.l[-2]/2 * sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains[-2:] = expr
            
            base_constrain = [0, 0]
            for i in range(self.n_rod - 1):
                base_constrain[0] += self.l[i] * sp.cos(self.q[i*3 + 2])
                base_constrain[1] += self.l[i] * sp.sin(self.q[i*3 + 2])
            base_constrain[0] += -self.l[self.n_rod-1]*sp.cos(self.q[3*(self.n_rod-1) + 2]) - self.l[-1]
            base_constrain[1] += -self.l[self.n_rod-1]*sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains.extend(base_constrain)
        
        Constrains = sp.Matrix(Constrains)
        
        return Constrains   
    
class ApproximateAlt(ApproximateModel):
    """docstring for CloseFourBar"""
    
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
            expr = self.q[(self.n_rod-1)*3:(self.n_rod-1)*3+2]
            expr[0] += -self.l[-1] - self.l[-2]/2 * sp.cos(self.q[3*(self.n_rod-1) + 2])
            expr[1] += - self.l[-2]/2 * sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains[-2:] = expr
            
            base_constrain = [0, 0]
            for i in range(self.n_rod - 1):
                base_constrain[0] += self.l[i] * sp.cos(self.q[i*3 + 2])
                base_constrain[1] += self.l[i] * sp.sin(self.q[i*3 + 2])
            base_constrain[0] += -self.l[self.n_rod-1]*sp.cos(self.q[3*(self.n_rod-1) + 2]) - self.l[-1]
            base_constrain[1] += -self.l[self.n_rod-1]*sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains.extend(base_constrain)
        
        Constrains = sp.Matrix(Constrains)
        
        return Constrains  
     
class ProjectAlt(ProjectModel):
    """docstring for ProjectClose"""
    
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
            expr = self.q[(self.n_rod-1)*3:(self.n_rod-1)*3+2]
            expr[0] += -self.l[-1] - self.l[-2]/2 * sp.cos(self.q[3*(self.n_rod-1) + 2])
            expr[1] += - self.l[-2]/2 * sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains[-2:] = expr
            
            base_constrain = [0, 0]
            for i in range(self.n_rod - 1):
                base_constrain[0] += self.l[i] * sp.cos(self.q[i*3 + 2])
                base_constrain[1] += self.l[i] * sp.sin(self.q[i*3 + 2])
            base_constrain[0] += -self.l[self.n_rod-1]*sp.cos(self.q[3*(self.n_rod-1) + 2]) - self.l[-1]
            base_constrain[1] += -self.l[self.n_rod-1]*sp.sin(self.q[3*(self.n_rod-1) + 2])
            Constrains.extend(base_constrain)
        
        Constrains = sp.Matrix(Constrains)
        
        return Constrains   
       
        
if __name__ == "__main__":
    
    import time, sys, os
    repo_dir = os.path.expanduser('~/Documents/course-projects/Constrained-Forward-Dynamic-Simulation-of-Multi-Links')
    sys.path.append(repo_dir + '/scripts')
    from scipy.integrate import solve_ivp,  odeint
    import matplotlib.pyplot as plt
    from solvers.fix_step_odes import *
    
    m = [1, 1, 1]
    l = [1, 4, 2.5, 3]
    g = [0, -9.81, 0, 0, -9.81, 0, 0, -9.81, 0]
    f = [0, 0, 5, 0, 0, 0, 0, 0, 0]
    
    #####################################Test Explict Model##########################################
    # y = np.append([0, 0.5, np.pi/2, 1.8765, 1.692, 0.3533, 3.3765, 1.192, -1.8767], np.zeros(9))
    # y = np.append([3.06161700e-17,  5.00000000e-01,  np.pi/2, 1.87648529e+00,  1.69195588e+00, 
    #                -5.92990441e+00,  3.37648529e+00,  1.19195588e+00,  1.06896358e+01], np.zeros(9))
    
    # y0 = [None, None, np.pi/2, None, None, None, None, None, None]
    # y_dot = [None, None, 0.1, None, None, None, None, None, None]
    # Demo = ExplictModel(m=m, l=l, close_chain=True)
    # pos, vel = Demo.initial_condition(y0, y_dot)
    # y = np.append(pos, vel)
    # # y = np.append(y, np.zeros(9))
    # t1 = time.time()
    # sol = solve_ivp(Demo.sim, [0, 10], y, method='DOP853', args=(f, g, None))
    # # sol = ode4(Demo.sim, np.linspace(0, 10, 5000), y, args=(f, g, None))
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()
    
    # print(Demo.lagrangian)
    
    ####################################Test Approximate Model######################################
    # y = np.append([3.06161700e-17,  5.00000000e-01,  np.pi/2, 1.87648529e+00,  1.69195588e+00, 
    #                -5.92990441e+00,  3.37648529e+00,  1.19195588e+00,  1.06896358e+01], 
    #               np.zeros(9))
    # k = np.tile([1e6], 8)
    # Demo = ApproximateModel(m=m, l=l, k=k, close_chain=True)    
    # # t0 = time.time()
    # # Demo.sim(t=0, y=y, f=f, g=g, c=None)
    # # print(time.time() - t0)
    # # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    # t1 = time.time()
    # sol = solve_ivp(Demo.sim, [0, 10], y, method='DOP853', args=(f, g, None))
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
    # y = np.append([3.06161700e-17,  5.00000000e-01,  np.pi/2, 1.87648529e+00,  1.69195588e+00, 
    #                -5.92990441e+00,  3.37648529e+00,  1.19195588e+00,  1.06896358e+01], 0)
    # # y = np.append([0, 0.5, np.pi/2, 1.8765, 1.692, 0.3533, 3.3765, 1.192, -1.8767], 0)
    # t0 = time.time()
    # Demo = ProjectModel(m, l, close_chain=True)
    # print(time.time() - t0)
    # # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    # t1 = time.time()
    # # # sol = ode4(Demo.sim, np.linspace(0, 5, 10000), y, args=(f, g, None))
    # sol = solve_ivp(Demo.sim, [0, 10], y, method='DOP853', args=(f, g, None))
    # print(sol.t.shape)
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()
    
    ####################################Test different constrains####################################
    # y = np.append([3.06161700e-17,  5.00000000e-01, np.pi/2, 1.08601471e+00, -6.79455882e-01, 
    #                -9.96782005e-01,  2.58601471e+00, -1.17945588e+00, -1.90835893e+00], 
    #               np.zeros(9))
    # k = np.tile([1e6], 8)
    # Demo = ExplictAlt(m, l, close_chain=True)
    # Demo = ApproximateAlt(m, l, k, close_chain=True)
    # Demo = ProjectAlt(m ,l, close_chain=True)
    # # t0 = time.time()
    # # Demo.sim(t=0, y=y, f=f, g=g, c=None)
    # # print(time.time() - t0)
    # # print(Demo.sim(t=0, y=y, f=f, g=g, c=None))
    # t1 = time.time()
    # sol = solve_ivp(Demo.sim, [0, 10], y, method='DOP853', args=(f, g, None))
    # print(sol.t.shape)
    # print(time.time() - t1)
    # plt.figure()
    # plt.plot(sol.t, sol.y[2])
    # plt.figure()
    # plt.plot(sol.t, sol.y[5])
    # plt.figure()
    # plt.plot(sol.t, sol.y[8])
    # plt.show()