# Aim

In this project, we aim to solve Fractional Differential Equations (FDE) on metric graphs using Physics-Informed Neural Networks (PINNs). We explore a wide variety of methods to optimize and reduce the errors of the output, utilizing $L^2$ errors on manufactured solutions to evaluate the effectiveness of the neural network.

---

## Parabolic Problem: Problem 1

### Governing Equation

$${^C D}_{0,t}^{\alpha} u(x,t) + f(x,t) + u(x,t)u_x(x,t) - 0.1 u_{xx}(x,t) = 0$$

For $0 < x < 1$ and $0 < t \leq 1$.

**Source Term & Conditions:**
* **Source Term:** $f(x,t) = -\frac{2 e^x t^{2-\alpha}}{\Gamma(3-\alpha)} - e^{2x} t^4 + 0.1 e^x t^2$
* **Initial Condition:** $u(x, 0) = 0$
* **Boundary Conditions:** $u(0, t) = t^2, \quad u(1, t) = e t^2$

**Exact Solution:**
For $\alpha \in (0, 1)$:
$$u(x,t) = e^x t^2$$

---

### Experimental Runs

#### Run 1: Soft Constraints
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Fractional Approximator** | L1 | **Strategy** | Dual |
| **Constraint Type** | Soft | **Causal Training** | Disabled |
| **Spatial / Temporal Points** | 100 / 100 | **NTK Balancing** | 200 iterations |
| **Iterations** | 10,000 | **RAD Sampling** | Disabled |
| **Architecture** | 60x4 | **L1 Points** | 0 |

![Run 1 Result] <img width="1829" height="687" alt="Screenshot 2026-02-18 211017" src="https://github.com/user-attachments/assets/9e1ccc9d-30fa-4476-ae24-b85d828b48f1" />


#### Run 2: Hard Constraints
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Fractional Approximator** | L1 | **Strategy** | Dual |
| **Constraint Type** | Hard | **Causal Training** | Disabled |
| **Spatial / Temporal Points** | 100 / 100 | **NTK Balancing** | 200 iterations |
| **Iterations** | 10,000 | **RAD Sampling** | Disabled |
| **Architecture** | 60x4 | **L1 Points** | 0 |

![Run 2 Result] <img width="1822" height="678" alt="Screenshot 2026-02-18 213437" src="https://github.com/user-attachments/assets/4fefa9b2-a6e5-42ef-9c8e-688978d21344" />
