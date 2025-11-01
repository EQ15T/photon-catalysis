from os import PathLike
from typing import Union
from pathlib import Path

import sympy as sp
import sympy.physics.quantum as sq

import numpy as np

from photon_catalysis.utils import kets_to_state_dict, state_to_tensor


def tensor_power(A, n):
    return sq.TensorProduct(*([A] * n))

def make_e3_mat(M) -> sp.Matrix:
    a = (sp.sympify(M - 2)**(1/sp.sympify(3)))
    V = sp.Matrix([
        [a] * M
    ])
    V = V.col_join(sp.Matrix.vstack(-sp.ones(M, M)))
    for i in range(1, M + 1):
        V[i, i - 1] *= -1
    return V / 24**(sp.sympify(1) / 3)

def esp(k, xs, return_poly=True) -> Union[sp.Expr, sp.Poly]:
    """
    Returns SymPy polynomial representing elementary symmetric polynomial of degree k in variables xs
    :param return_poly: if True, casts expression to polynomial in suitable field extension
    :return:
    """
    def esp_rec(k, xs) -> sp.Expr:
        if k == 0:
            return sp.sympify(1)
        if k == 1:
            return sum(xs, sp.sympify(0))
        s = sp.sympify(0)
        for j in range(len(xs) - k + 1):
            s += xs[j] * esp_rec(k - 1, xs[j + 1:])
        return s

    r = esp_rec(k, xs)
    if not return_poly:
        return r
    return sp.Poly(r.expand(), extension=True)


def get_polysys3(M, N, target: np.ndarray) -> list[sp.Poly]:
    """
    :param M: Number or target modes
    :param N: Number of intermediate modes
    :param target: Target homogeneous polynomial
    :return: List of polynomials of degree 3. Solving all of them simultaneously (as a system) corresponding to the solution
    of decomposition e3(v1*x, ..., vN*x) = p(x1, ..., xM), where x=[x1, ..., xM] and vi = [vi1, ..., viM].
    """
    U = make_e3_mat(N)
    r = U.rows
    e3 = sum([tensor_power(U.row(i), 3) for i in range(r)], sp.zeros(1, N**3))

    ys = sp.Matrix(sp.symbols(f'x_1:{N + 1}'))
    assert ((e3 * tensor_power(ys, 3))[0, 0] == esp(3, ys, False).expand().simplify())

    vars = N * M
    vs = sp.symbols(f'v1:{vars + 1}')
    V = sp.Matrix(N, M, vs)


    def get_ket(i, dim):
        r = sp.zeros(dim, 1)
        r[i] = 1
        return r

    polys = []
    for j1 in range(M):
        for j2 in range(M):
            for j3 in range(M):
                s = 0
                for i1 in range(N):
                    for i2 in range(N):
                        for i3 in range(N):
                            c = e3 * sq.TensorProduct(get_ket(i1, N), get_ket(i2, N), get_ket(i3, N))
                            c = c[0, 0]
                            s += c * V[i1, j1] * V[i2, j2] * V[i3, j3]
                # val = get_p_val((j1, j2, j3))
                val = target[j1, j2, j3]
                polys.append(sp.Poly(s - val, *vs, domain='QQ'))
    return polys


def to_msolve(fname, polysys: list[sp.Poly]):
    """
    Export given polynomial system to the file in msolve format
    :param fname: Filename
    :param polysys: Polynomial system
    """
    with open(fname, 'w') as f:
        vars = polysys[0].gens
        for p in polysys:
            assert(set(vars) == set(p.gens))
        f.write(','.join(map(lambda x: str(x), vars)) + '\n')
        f.write('0\n') # field characteristic
        for p in polysys:
            f.write(sp.printing.mathematica.mathematica_code(p.expr) + ',\n')


def polysys2mat(polysys: list[sp.Poly]) -> sp.Matrix:
    res = sp.Matrix.zeros(len(polysys), 1)
    for i in range(len(polysys)):
        res[i, 0] = polysys[i].expr
    return res

def polysys2func(polysys: list[sp.Poly]):
    vars = list(polysys[0].gens)
    return sp.lambdify(vars, polysys2mat(polysys), 'numpy')

def polysys2jac(polysys: list[sp.Poly]):
    vars = list(polysys[0].gens)
    mat = polysys2mat(polysys)
    J = sp.Matrix.zeros(len(polysys), len(vars))
    for r in range(len(polysys)):
        for c in range(len(vars)):
            J[r, c] = sp.diff(mat[r, 0], vars[c])
    return sp.lambdify(vars, J, 'numpy')

def optimize_3(N: int, target: np.ndarray) -> sp.Matrix:
    M = target.shape[0]
    polysys = get_polysys3(M, N, target)
    vars = list(polysys[0].gens)
    func = polysys2func(polysys)
    jac = polysys2jac(polysys)

    x = None
    for s in range(100):
        x = np.random.randn(len(vars))
        n = None
        for i in range(500):
            J = jac(*x)
            f = np.squeeze(func(*x))
            r = np.linalg.pinv(J) @ f
            x = x - r
            n = np.linalg.norm(f)
            if n < 0.001:
                break
        if n < 0.001:
            print(f'Converged on iteration {s+1}')
            break
    print(x)
    print(f'||f|| = {np.linalg.norm(func(*x)):.5f}')
    V = sp.Matrix(N, M, x)
    xs = sp.Matrix(V.cols, 1, sp.symbols(f'x1:{V.cols+1}'))
    poly = esp(3, V * xs, True)
    print(poly)
    return V


def main():
    states = {
        'psi_2' : kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3)]),
        'psi_5' : kets_to_state_dict([(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]),
        'psi_9' : kets_to_state_dict([(3, 0, 0, 0), (0, 2, 1, 0), (0, 1, 2, 0), (0, 0, 0, 3)]),
        'R4' : kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 1, 1)]),
        'R5' : kets_to_state_dict([(2, 1, 0), (0, 2, 1)]),
        'K3' : {
             (3, 0, 0, 0) : 1,
             (2, 1, 0, 0) : 1,
             (2, 0, 1, 0) : 1,
             (2, 0, 0, 1) : 1,
             (1, 1, 1, 0) : -1,
             (1, 1, 0, 1) : -1,
             (1, 0, 1, 1) : -1,
             (0, 1, 1, 1) : -1 }
    }
    states = {k: np.asarray(state_to_tensor(v)) for k, v in states.items()}

    basepath = Path('/home/andrew/Documents/Sorbonne/M1_2/QPh/cat-states-gaussian-transformation/alg_methods/t/')

    for name, state in states.items():
        print()
        print(name)
        path3 = basepath / (name + '_3.txt')
        path4 = basepath / (name + '_4.txt')
        M = state.shape[0]
        to_msolve(basepath / path3, get_polysys3(M, M, state))
        to_msolve(basepath / path4, get_polysys3(M, M+1, state))
        optimize_3(M+1, state)

    # optimize_3(4, np.asarray(state_to_tensor(states['psi_2'])))
    # optimize_3(4, np.asarray(state_to_tensor(states['psi_5'])))
    # optimize_3(5, np.asarray(state_to_tensor(states['psi_9'])))

if __name__ == '__main__':
    main()
    # V = sp.Matrix(4, 3, [-5.24415029e-03, -2.98452803e-03, -4.97494597e-03,  5.24414925e-03,
    #      2.98452798e-03,  4.97494495e-03, -1.96951332e+04, -7.58604844e+04,
    #     -8.06389656e+04, -6.49996707e+04, -5.88097726e+04, -1.72812110e+04])
    # print(make_poly(V))