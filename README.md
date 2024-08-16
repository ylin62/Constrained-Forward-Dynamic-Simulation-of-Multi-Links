# Constrained Forward Dynamic Simulation of Multi-Links 

## Introduction
This project provided classes of modeling open and closed loop uniformly distributed mass serial links. Modeling based on Lagrangian methods with constrained generalized coordinates. Equations of Motion (EOM) are computed sybolically with sympy. EOMs can be manipulated into three different forms of state space models:

1. Eliminate Lagrangian multipliers.
2. Approximate constrains as springs.
3. Project governing equations to null space.

Provided initial condition solver, constrained force, torque and motion can be calculated with ODE solvers. 

Comapared performance of different methods and ode solvers in terms of computational cost and accuracy. A brief presentation including modeling details and results can be found [here](Serial_links.pdf).

[Double Pendulum](/scripts/ExplicitModel_Double_Pendulum.ipynb) | [Triple Pendulum](/scripts/ExplicitModel_Triple_Pendulum.ipynb) | [Quadruple Pendulum](/scripts/ExplicitModel_Quadruple_Pendulum.ipynb) | [Four-bar](/scripts/ExplicitModel_Fourbar.ipynb) | [Deca Pendulum](/scripts/ExplicitModel_Deca_Pendulum.ipynb)
--------------- | --------------- | ------------------ | -------- | -------------
![](imgs/DoublePendulum.gif) | ![](imgs/TriplePendulum.gif) | ![](imgs/QuadruplePendulum.gif) | ![](imgs/Fourbar.gif) | ![](imgs/DecaPendulum.gif)

Jupyter notebook demo with interactive animation in [scripts](/scripts)

Add gifs and video demos in [imgs](/imgs)
