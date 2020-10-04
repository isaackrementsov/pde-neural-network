# PDE Neural Network
Neural network to solve the 1d Wave Equation without training data
### How it will work
For a neural network to solve the wave equation PDE, it will have generate a hypothesis function h that satisfies
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial^2&space;h}{\partial&space;t^2}&space;-&space;v^2\frac{\partial^2&space;h}{\partial&space;x^2}&space;=&space;0" title="\frac{\partial^2 h}{\partial t^2} - v^2\frac{\partial^2 h}{\partial x^2} = 0" /></p>

as well as the boundary conditions listed in the original problem. The loss function can therefore be written as
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?L(x,t)&space;=&space;(\frac{\partial^2&space;h}{\partial&space;t^2}&space;-&space;v^2\frac{\partial^2&space;h}{\partial&space;x^2})^2&space;&plus;&space;L_{bd}&space;&plus;&space;L_{bn}"/></p>
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?L_{bd}(x,t)&space;=&space;\begin{cases}&space;h(x,t)^2&space;&&space;x&space;=&space;0&space;\text{&space;or&space;}&space;x&space;=&space;L&space;\\&space;0&space;&&space;x&space;\ne&space;0,L&space;\end{cases}"/></p>
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?L_{bn}&space;=&space;\begin{cases}&space;(\frac{\partial&space;h}{\partial&space;t}\rvert_{(x,0)}&space;-&space;g(x))^2&space;&plus;&space;(h(x,0)&space;-&space;f(x))^2&space;&&space;t&space;=&space;0&space;\\&space;0&space;&&space;t&space;\ne&space;0&space;\end{cases}"/></p>
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?\text{Where&space;}&space;f(x)&space;\text{&space;and&space;}&space;g(x)&space;\text{&space;are&space;the&space;boundary&space;conditions&space;for&space;}&space;u(x,0)&space;\text{&space;and&space;}&space;\frac{\partial&space;u}{\partial&space;t}\rvert_{(x,0)}"/></p>
This makes the total loss
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\frac{1}{N_xN_t}\sum\limits_{i&space;=&space;0}^{N_x}\sum\limits_{j&space;=&space;0}^{N_t}&space;L(x_i,&space;t_j)&space;\text{&space;,&space;where&space;}&space;x&space;\text{&space;and&space;}&space;t&space;\text{&space;are&space;discretized&space;into&space;a&space;}&space;N_x\times&space;N_t&space;\text{&space;matrix,&space;}&space;D"/></p>
<p style="text-align: center"><img src="https://latex.codecogs.com/gif.latex?\text{The&space;neural&space;network's&space;goal&space;is&space;to&space;find&space;parameters&space;}&space;b^*,&space;w^*&space;\text{&space;such&space;that&space;}&space;(b^*,&space;w^*)&space;=&space;\arg\min&space;L(D;&space;b,&space;w)"/></p>
