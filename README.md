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



#### Run 3: Soft Constraints
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Fractional Approximator** | L21-σ | **Strategy** | Dual |
| **Constraint Type** | Soft | **Causal Training** | Disabled |
| **Spatial / Temporal Points** | 100 / 100 | **NTK Balancing** | 200 iterations |
| **Iterations** | 10,000 | **RAD Sampling** | Disabled |
| **Architecture** | 60x4 | **L1 Points** | 0 |

![Run 3 Result] <img width="1600" height="590" alt="image" src="https://github.com/user-attachments/assets/184b24d9-f8f9-4540-9acf-06db5f6f80cc" />



#### Run 4: Hard Constraints
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Fractional Approximator** | L21-σ | **Strategy** | Dual |
| **Constraint Type** | Hard | **Causal Training** | Disabled |
| **Spatial / Temporal Points** | 100 / 100 | **NTK Balancing** | 200 iterations |
| **Iterations** | 10,000 | **RAD Sampling** | Disabled |
| **Architecture** | 60x4 | **L1 Points** | 0 |



## Parabolic Problem: Problem 2

### Governing Equation

$$^C D_{0,t}^{\alpha} u(x,t) + u(x,t)u_x(x,t) - 0.1\, u_{xx}(x,t) = f(x,t)$$

For $0 < x < 1$ and $0 < t \leq 1$.

**Source Term & Conditions:**

* **Source Term:**

$$f(x,t) = \frac{2\,t^{2-\alpha}}{\Gamma(3-\alpha)}\sin(\pi x)\,e^{-k(x-x_0)^2} - t^4\sin^2(\pi x)\,e^{-2k(x-x_0)^2}\Big[\pi\cos(\pi x) - 2k(x-x_0)\sin(\pi x)\Big] - 0.1\,t^2\,e^{-k(x-x_0)^2}\Big[-\pi^2\sin(\pi x) - 4k\pi\cos(\pi x)(x-x_0) + \sin(\pi x)\big(4k^2(x-x_0)^2 - 2k\big)\Big]$$

with $k = 20$, $x_0 = 0.5$

* **Initial Condition:** $u(x, 0) = 0$

* **Boundary Conditions:** $u(0, t) = 0, \quad u(1, t) = 0$

**Exact Solution:**

For $\alpha \in (0, 1)$:

$$u(x,t) = t^2 \sin(\pi x)\, e^{-k(x - x_0)^2}$$

<img width="1606" height="456" alt="image" src="https://github.com/user-attachments/assets/caa8e030-4160-410d-8d76-d30487ba4e7f" />

<img width="1574" height="464" alt="image" src="https://github.com/user-attachments/assets/391cc8de-4ab1-47a8-9b2a-a4fd0d69f03b" />







