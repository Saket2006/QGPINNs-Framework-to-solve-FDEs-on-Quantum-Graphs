# QGPINNs Framework

This repository contains the source code for the QGPINNs framework. The QGPINNs framework allows users to obtain numerical solutions to fractional differential equations defined on metric graphs. We primarily focus on elliptic and parabolic type fractional differential equations, however the framework is flexible enough to accommodate a significantly larger set of equations. A detailed paper on the methods and techniques used in this framework are present in......

## Framework Architecture

To ensure that the engine remains modular and is adaptable to a wide variety of problems, we separate the engine containing the neural network architecture and the run script which defines the governing physics and metric graph topology.  The QGPINNs engine can be accessed [here](Engine/QGPINNs_Engine). We also provide all the run scripts used in the paper. All run scripts used in Section 3 can be accessed [here](Section_3_Run_Scripts). The numerical section details problems based on real world topologies, the corresponding run scripts can be found [here](Section_5_Run_Scripts).



