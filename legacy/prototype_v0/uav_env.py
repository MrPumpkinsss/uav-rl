
import numpy as np
import cvxpy as cp

def solve_P2(X_bin,ppl_delay, h_matrix, config):
    N, L = X_bin.shape
    cl = np.array(config['cl'])
    ol = np.array(config['ol'])
    fn = np.array(config['fn'])
    pn = np.array(config['pn'])
    Bmax = config['Bmax']
    k0 = config['k0']
    Phov = config['Phov']
    N0 = config['N0']
    Emax = np.array(config['Emax'])

    B = cp.Variable(N, nonneg=True)

    tcmp = np.zeros(N)
    ecmp = np.zeros(N)
    for n in range(N):
        for l in range(L):
            if X_bin[n, l] == 1:
                tcmp[n] += cl[l] / fn[n]
                ecmp[n] += k0 * fn[n]**2 * cl[l]

    tcom_expr = []
    ecom_expr = []
    for l in range(L - 1):
        for n in range(N):
            if X_bin[n, l] == 1:
                for n2 in range(N):
                    if X_bin[n2, l + 1] == 1:
                        # 替代非法除法：使用 inv_pos() 保证符合 DCP 规则
                        r = B[n] * cp.log1p((pn[n] * h_matrix[n, n2]) / N0)
                        tcom_l = ol[l] * cp.inv_pos(r)
                        ecom_l = pn[n] * tcom_l
                        tcom_expr.append(tcom_l)
                        ecom_expr.append(ecom_l)

    tcom_total = cp.sum(tcom_expr)
    ecom_total = cp.sum(ecom_expr)
    Thov = cp.sum(tcmp) + tcom_total
    ehov = Phov * Thov

    constraints = [cp.sum(B) <= Bmax]
    for n in range(N):
        constraints.append(ecmp[n] + ecom_total + ehov <= Emax[n])

    Q = config['alpha'] * ppl_delay + config['beta'] * Thov
    prob = cp.Problem(cp.Minimize(Q), constraints)
    prob.solve(solver=cp.SCS)

    if prob.status not in ["infeasible", "unbounded"]:
        return -Q.value, B.value
    else:
        return -np.inf, np.zeros(N)

class UAVDeploymentEnv:
    def __init__(self, config):
        self.config = config
        self.N = config['N']
        self.L = config['L']
        self.state_dim = self.N * self.N

    def reset(self):
        self.h = np.random.uniform(0.1, 1.0, size=(self.N, self.N))
        return self.h.flatten()

    def step(self, action_bin, ppl_delay):
        # X_bin = action.reshape((self.N, self.L))
        reward, _ = solve_P2(action_bin, ppl_delay, self.h, self.config)
        self.h = np.random.uniform(0.1, 1.0, size=(self.N, self.N))
        return self.h.flatten(), reward
