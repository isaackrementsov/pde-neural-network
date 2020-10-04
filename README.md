# PDE Neural Network
Neural network to solve the 1d Wave Equation without training data
### How it will work
For a neural network to solve the wave equation PDE, it will have generate a hypothesis function h that satisfies
$$ \frac{\partial^2 h}{\partial t^2} - v^2\frac{\partial^2 h}{\partial x^2} = 0 $$
as well as the boundary conditions listed in the original problem. The loss function can therefore be written as
$$ L(x,t) = (\frac{\partial^2 h}{\partial t^2} - v^2\frac{\partial^2 h}{\partial x^2})^2 + L_{bd} + L_{bn} $$
<br>
$$ L_{bd}(x,t) = \begin{cases}
h(x,t)^2 & x = 0 \text{ or } x = L \\
0 & x \ne 0,L
\end{cases} $$
<br>
$$ L_{bn} = \begin{cases}
(\frac{\partial h}{\partial t}\rvert_{(x,0)} - g(x))^2 + (h(x,0) - f(x))^2 & t = 0 \\
0 & t \ne 0
\end{cases}$$
$$ \text{Where } f(x) \text{ and } g(x) \text{ are the boundary conditions for } u(x,0) \text{ and } \frac{\partial u}{\partial t}\rvert_{(x,0)} $$
<br>
This makes the total loss
$$ L = \frac{1}{N_xN_t}\sum\limits_{i = 0}^{N_x}\sum\limits_{j = 0}^{N_t} L(x_i, t_j) \text{ , where } x \text{ and } t \text{ are discretized into a } N_x\times N_t \text{ matrix, } D $$
$$ \text{The neural network's goal is to find parameters } b^*, w^* \text{ such that } (b^*, w^*) = \arg\min L(D; b, w) $$
