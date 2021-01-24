## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/ylin62/Constrained-Forward-Dynamic-Simulation-of-Multi-Links/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

# Constrained Forward Dynamic Simulation of Multi-Links 

## Introduction
This project provided classes of modeling open and closed loop uniformly distributed mass serial links. Modeling based on Lagrangian methods with constrained generalized coordinates. Equations of Motion (EOM) are computed sybolically with sympy. EOMs can be manipulated into three different forms of state space models:

1. Eliminate Lagrangian multipliers.
2. Approximate constrains as springs.
3. Null space.

Provided initial condition solver, constrained force, torque and motion can be calculated with ODE solvers. 

Comapare performance of different methods and ode solvers in terms of computational cost and accuracy. A brief presentation including modeling details and results can be found [here](Serial_links.pdf).

[Double Pendulum](/scripts/ExplicitModel_Double_Pendulum.ipynb) | [Triple Pendulum](/scripts/ExplicitModel_Triple_Pendulum.ipynb) | [Quadruple Pendulum](/scripts/ExplicitModel_Quadruple_Pendulum.ipynb) | [Four-bar](/scripts/ExplicitModel_Fourbar.ipynb) | [Deca Pendulum](/scripts/ExplicitModel_Deca_Pendulum.ipynb)
--------------- | --------------- | ------------------ | -------- | -------------
![](imgs/DoublePendulum.gif) | ![](imgs/TriplePendulum.gif) | ![](imgs/QuadruplePendulum.gif) | ![](imgs/Fourbar.gif) | ![](imgs/DecaPendulum.gif)

Jupyter notebook demo with interactive animation in [scripts](/scripts)

Add gifs and video demos in [imgs](/imgs)
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ylin62/Constrained-Forward-Dynamic-Simulation-of-Multi-Links/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
