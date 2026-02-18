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

![Run 1 Result] <img width="1591" height="592" alt="image" src="https://github.com/user-attachments/assets/5b6f8a7d-f1bf-4525-ab95-9846e021dd65" />



#### Run 2: Hard Constraints
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Fractional Approximator** | L1 | **Strategy** | Dual |
| **Constraint Type** | Hard | **Causal Training** | Disabled |
| **Spatial / Temporal Points** | 100 / 100 | **NTK Balancing** | 200 iterations |
| **Iterations** | 10,000 | **RAD Sampling** | Disabled |
| **Architecture** | 60x4 | **L1 Points** | 0 |

![Run 2 Result] <img width="1586" height="602" alt="image" src="https://github.com/user-attachments/assets/9cf74608-d08f-46cb-a985-13b1b0e8877e" />

