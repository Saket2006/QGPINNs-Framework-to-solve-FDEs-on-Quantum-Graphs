# FDE-on-Metric-Graphs-Using-PINNs

## Pre-view
This file contains various results solved using our PINN engine\
We have taken a relatively complex manufactured solution to test the strength of the engine\
We are finding the error rate of this manufactured solution on various graphs\
We are also comparing different methods of weight estimation and mesh creation\

## Manufactured Solution 1

We use the manufactured solution:\
                u(x)=cos(2πx/L) where L represent the edge length
                

1)To set a base-line, we test our model on an interval of length 1:-\
\
        a)Gradient Ratio with Graded Mesh(r=1.75) and point density 800 (5000 iterations):\
        <img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/9e4429eb-570f-4b8b-be49-275b9a739563" />\
         --- Error Report ---\
         Edge 0 (0, 1): Rel L2 = 7.4256e-07\
         \
         \
        b)Gradient Ratio with Adaptive Mesh and point density 800 (5000 iterations):\
        <img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/52ca1b52-5107-4a3c-adca-5e7e8eae99a0" />\
        --- Error Report ---\
        Edge 0 (0, 1): Rel L2 = 2.6116e-06\
        \
        \
        c)BDMM with Adaptive Mesh and point density 800 (5000 iterations):\
        <img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/67be0c72-da36-4dbc-b05f-66a4432f6a48" />\
        --- Error Report ---\
        Edge 0 (0, 1): Rel L2 = 1.0098e-04\
        \
        \
        d)BDMM with Graded Mesh(r=1.75) and point density 800 (5000 iterations):\
        <img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/3d325a66-2dbc-42dd-a145-8a6fd50035ee" />\
        --- Error Report ---\
        Edge 0 (0, 1): Rel L2 = 6.7317e-06\




        

        


