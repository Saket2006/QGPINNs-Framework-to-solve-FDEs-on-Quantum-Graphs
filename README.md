# FDE-on-Metric-Graphs-Using-PINNs

## Pre-view
This file contains various results solved using our PINN engine\
We have taken a relatively complex manufactured solution to test the strength of the engine\
We are finding the error rate of this manufactured solution on various graphs\
We are also comparing different methods of weight estimation and mesh creation\
The manufacured solution used is:\
        u(x)=cos(2πx/L) + 0.4*cos(10πx/L) + 0.2*cos(18πx/L)\
        where L is the length of the current edge

## Result on an interval

To set a base-line, we test our model on an interval of length 1
