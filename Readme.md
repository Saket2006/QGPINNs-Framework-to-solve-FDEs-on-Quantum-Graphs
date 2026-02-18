
# Aim

In this project we aim to solve FDE on metric graphs using PINNs. We explore a wide variety of methods to optimize and reduce the errors of the output. We find the L2 errors on manufactured solutions to understand the effectiveness of our neural network.

## Parabolic Problem:

### Problem 1:

$${^C D}_{0,t}^{\alpha} u(x,t) + f(x,t) + u(x,t)u_x(x,t) - 0.1 u_{xx}(x,t) = 0$$

$0 < x < 1, \quad 0 < t \leq 1$

**Source Term & Conditions:**

- $f(x,t) = -\frac{2 e^x t^{2-\alpha}}{\Gamma(3-\alpha)} - e^{2x} t^4 + 0.1 e^x t^2$
    
- $u(x, 0) = 0$
    
- $u(0, t) = t^2, \quad u(1, t) = e t^2$
    

**Exact Solution:**

For $\alpha \in (0, 1)$:

$$u(x,t) = e^x t^2$$

**Run 1 Parameters**: 
Fractional Approximator:L1
Constraint type: soft
Spatial points:100
Temporal points:100
L1 points:0
Iterations:10000
Architecture:60x4
Strategy: Dual
Causal Training: Disabled
NTK balancing: 200 iterations
RAD sampling: Disabled

![[Screenshot 2026-02-18 211017.png]]

**Run 2 Parameters**: 
Fractional Approximator:L1
Constraint type: hard
Spatial points:100
Temporal points:100
L1 points:0
Iterations:10000
Architecture:60x4
Strategy: Dual
Causal Training: Disabled
NTK balancing: 200 iterations
RAD sampling: Disabled

![[Pasted image 20260218213441.png]]

