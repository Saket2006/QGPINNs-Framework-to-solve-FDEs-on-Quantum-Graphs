from pinn_graph_engine import GraphPINNSolver
import numpy as np

class Graph:
    def __init__(self):
        self.edges = [

            (0, 1, 1.0)
        ]
        self.nodes = [0, 1]


class MyPhysics:
    def __init__(self, alpha=1.7, beta=0.3):
        self.alpha, self.beta = alpha, beta

    def F(self, x, u, du, d_beta_u, f_target):
        return f_target - (du + u + d_beta_u)

    def get_f_target(self, x_np, L, Da, Db):

        #u = cos(2πx/L)
        k1, k2, k3 = 2 * np.pi / L, 10 * np.pi / L, 18 * np.pi / L

        uex = np.cos(k1 * x_np) )
        duex = -k1 * np.sin(k1 * x_np) 
        f_target = (Da @ duex) + duex + uex + (Db @ uex)
        return f_target


def benchmark(x, L):
    k1, k2, k3 = 2 * np.pi / L, 10 * np.pi / L, 18 * np.pi / L
    return np.cos(k1 * x) 

if __name__ == "__main__":
    bc_types = {0: "dirichlet", 1: "dirichlet"}
    bc_values = {0: 1.0, 1: 1.0}

    solver = GraphPINNSolver(
        Graph(),
        MyPhysics(),
        strategy="bdmm",
        mesh_type="graded",
        pts_per_unit=800,
        grading_factor=1.75,
        bc_types=bc_types,
        bc_values=bc_values
    )

    solver.plot_graph_topology()
    solver.train(epochs=5000, use_lbfgs=True, adaptive_every=1000)
    solver.report_l2(benchmark)
    solver.post_process(benchmark)
