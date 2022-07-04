
- $a$: A scalar
- $\bold{a}$: A vector
- $\bold{A}$: A matrix
- $\bold{A}_{i,:}$: Row $i$ of matrix $\bold{A}$
- $\bold{A}_{:,j}$: Column $j$ of matrix $\bold{A}$
- $\bold{W^{(l)}}$: weights connect from (l-1)-th layer to l-th layer
- $\bold{b}^{(l)}$: bias connect from (l-1)-th layer to l-th layer
- $w_{ij}^{(l)}$: weight connects from i-th neuron of (l-1)-th layer to jth neuron lth layer
- $z_{i}^{(l)}$: pre-activated output of neuron i-th of l-th layer
- $a_{i}^{(l)}$: activated output of neuron i-th of l-th layer
- $d^{(l)}$: number of neuron (dimension) of l-th layer.
- $\bold{z}^{(l)}$: pre-activated output vector of l-th layer.  $\bold{z}^{(l)} = \begin{bmatrix} z_1^{(l)} & z_2^{(l)} & \dots & z_{d^{(l)}}^{(l)}\end{bmatrix}^T$
- $\bold{a}^{(l)}$: activated output vector of l-th layer. $\bold{a}^{(l)} = \begin{bmatrix} a_1^{(l)} & a_2^{(l)} &\dots & a_{d^{(l)}}^{(l)}\end{bmatrix}^T$

- $\mathcal{L}$: loss function

- $\odot$: Element-wise product of two vectors

- $\bold{x}=\bold{a}^{(0)}$: input of neural network


$$\bold{W}^{(l)}=\begin{bmatrix} w_{11}^{(l)} & w_{12}^{(l)} & w_{13}^{(l)} & \dots & w_{1d^{(l)}}^{(l)} \\ w_{21}^{(l)} & w_{22}^{(l)} & w_{23}^{(l)} & \dots & w_{2d^{(l)}}^{(l)} \\ \vdots & \thickspace & \thickspace & \thickspace & \vdots \\ w_{d^{(l-1)}1}^{(l)} & w_{d^{(l-1)}2}^{(l)} & w_{d^{(l-1)}3}^{(l)} & \dots & w_{d^{(l-1)}d^{(l)}}^{(l)}\end{bmatrix} \in \mathbb{R}^{d^{(l-1)} \times d^{(l)}}$$

$$\bold{b}^{(l)}=\begin{bmatrix} b_1^{(l)} \\  b_2^{(l)} \\ \vdots \\ b_{d^{(l)}}^{(l)} \end{bmatrix}$$

$$z_j^{(l)}=\sum_{i=1}^{d^{(l-1)}}w_{ij}^{(l)}a_i^{(l-1)}+b_j^{(l)}=\Big(\bold{W}_{:,i}\Big)^T \bold{a}^{(l-1)} + b_j^{(l)}$$

$$\bold{z}^{(l)}=\Big(\bold{W}^{(l)}\Big)^T\bold{a}^{(l-1)} + \bold{b}^{(l)}, l=1, \dots, L$$

$$\bold{a}^{(l)}=\bold{f}(\bold{z}^{(l)})$$

$$\mathcal{L}=\bold{g}\Big(\bold{a}^{(L)}\Big)$$

We need to calculate partial derivatives of $\mathcal{L}$ with respect to all $\bold{W}^{(l)}$ and $\bold{b}^{(l)}, l=1, \dots, L$

$$\dfrac{\partial \mathcal{L}}{w_{ij}^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(l)}}\dfrac{\partial z_j^{(l)}}{w_{ij}^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(l)}}a_i^{(l-1)}$$

$$\dfrac{\partial \mathcal{L}}{\partial b_j^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(l)}}\dfrac{\partial z_j^{(l)}}{\partial b_j^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(l)}}$$

We set:

$$\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(l)}}=e_j^{(l)}$$

At output layer ($L$-th layer):

$$\dfrac{\partial \mathcal{L}}{w_{ij}^{(L)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(L)}}\dfrac{\partial z_j^{(L)}}{w_{ij}^{(L)}}=\dfrac{\partial \mathcal{L}}{\partial z_{j}^{(L)}}a_i^{(L-1)}=\dfrac{\partial\mathcal{L}}{\partial a_j^{(L)}}\dfrac{\partial a_j^{(L)}}{\partial z_j^{(L)}}a_i^{(L-1)}=\dfrac{\partial\mathcal{L}}{\partial a_j^{(L)}}\bold{f}^{\prime}\Big(z_j^{(L)}\Big)a_i^{(L-1)}$$

$$\bold{e}^{(L)}=\begin{bmatrix} \dfrac{\partial\mathcal{L}}{\partial a_1^{(L)}}\bold{f}^{\prime}\Big(z_1^{(L)}\Big) & \dfrac{\partial\mathcal{L}}{\partial a_2^{(L)}}\bold{f}^{\prime}\Big(z_2^{(L)}\Big) & \dots & \dfrac{\partial\mathcal{L}}{\partial a_{d^{(L)}}^{(L)}}\bold{f}^{\prime}\Big(z_{d^{(L)}}^{(L)}\Big) \end{bmatrix}^T=\dfrac{\partial \mathcal{L}}{\partial \bold{a}^{(L)}} \odot \bold{f}^{\prime}(\bold{z}^{(L)})$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(L)}}=\bold{a}^{(L-1)}\Big(\bold{e}^{(L)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(L)}}=\bold{e}^{(L)}$$

At l-th layer ($1 \leq l < L$):

$$\dfrac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}}\dfrac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}} a_i^{(l-1)}=e_j^{(l)}a_i^{(l-1)}$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(l)}}=\bold{a}^{(l-1)}\Big(\bold{e}^{(l)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial b_j^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}}\dfrac{\partial z_j^{(l)}}{\partial b_j^{(l)}}=\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}} = e_j^{(l)}$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(l)}}=\bold{e}^{(l)}$$

We consider 
$\mathcal{L} = \mathcal{L}\Big(\bold{z}^{(l+1)}\Big)=\mathcal{L}\Big(z_1^{(l+1)}, z_2^{(l+1)}, \dots, z_{d^{(l+1)}}^{(l+1)}\Big)$:

$$\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}}=\sum_{k=1}^{d^{(l+1)}}\dfrac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} \dfrac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}}=\sum_{k=1}^{d^{(l+1)}} \dfrac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} \dfrac{\partial z_k^{(l+1)}}{\partial a_j^{(l)}} \dfrac{\partial a_j^{(l)}}{\partial z_j^{(l)}}=\sum_{k=1}^{d^{(l+1)}}\dfrac{\partial \mathcal{L}}{\partial z_k^{(l+1)}}w_{jk}^{(l+1)}\bold{f}^{\prime}\Big(z_j^{(l)}\Big)$$

$$e_j^{(l)}=\dfrac{\partial \mathcal{L}}{\partial z_j^{(l)}}=\sum_{k=1}^{d^{(l+1)}}e_k^{(l+1)}w_{jk}^{(l+1)}\bold{f}^{\prime}\Big(z_j^{(l)}\Big)=\bold{f}^{\prime}(z_j^{(l)})\bold{W}_{j,:}^{(l+1)}\bold{e}^{(l+1)}$$

In matrix form:

$$\bold{e}^{(l)}=\bold{W}^{(l+1)}\bold{e}^{(l+1)}\odot\bold{f}^{\prime}(\bold{z}^{(l)})$$

- Summary:

Step 1:

Compute forward pass:

$\bold{a}^{(0)}=\bold{x}$

$l=1, \dots, L$:

$$\bold{z}^{(l)}=\Big(\bold{W}^{(l)}\Big)^T \bold{a}^{(l-1)} + \bold{b}^{(l)}$$


$$\bold{a}^{(l)}=\bold{f}\Big(\bold{z}^{(l)}\Big)$$

Step 2:

Compute $\mathcal{L}=\bold{g}\Big(\bold{a}^{(L)}\Big)$

Compute backward pass:

$l=L$: $$\bold{e}^{(L)}=\dfrac{\partial \mathcal{L}}{\partial \bold{a}^{(L)}} \odot \bold{f}^{\prime}(\bold{z}^{(L)})$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(L)}}=\bold{a}^{(L-1)}\Big(\bold{e}^{(L)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(L)}}=\bold{e}^{(L)}$$

$l=L-1, \dots, 1$:

$$\bold{e}^{(l)}=\bold{W}^{(l+1)}\bold{e}^{(l+1)}\odot\bold{f}^{\prime}(\bold{z}^{(l)})$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(l)}}=\bold{a}^{(l-1)}\Big(\bold{e}^{(l)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(l)}}=\bold{e}^{(l)}$$


Step 3:

Update neural network weights (example: SGD):

$l=1, \dots, L$:

$$\bold{W}^{(l)} \leftarrow \bold{W}^{(l)} - \alpha \dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(l)}}$$

$$\bold{b}^{(l)} \leftarrow \bold{b}^{(l)} - \alpha \dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(l)}}$$

For Minibatch-Gradient Descent: Batch size $N$

Call: 
$$\bold{X}=\bold{A}^{(0)}=\begin{bmatrix} \bold{x}_1 \vert \bold{x}_2 \vert \dots \vert \bold{x}_n \vert \dots \vert  \bold{x}_N \end{bmatrix} \in \mathbb{R}^{d^{(0)} \times N}$$


$$Z^{(l)}=\begin{bmatrix} \bold{z}_1^{(l)} \vert \bold{z}_2^{(l)} \vert \dots \vert \bold{z}_n^{(l)} \vert \dots \vert \bold{z}_N^{(l)} \end{bmatrix}$$

$$A^{(l)}=\bold{f}\Big(\bold{Z}^{(l)}\Big)=\begin{bmatrix} \bold{a}_1^{(l)} \vert \bold{a}_2^{(l)} \vert \dots \vert \bold{a}_n^{(l)} \vert \dots \vert \bold{a}_N^{(l)} \end{bmatrix}$$

$$\bold{B}^{(l)}=\underbrace{\begin{bmatrix} \bold{b}^{(l)} \vert \bold{b}^{(l)} \vert \dots \vert \bold{b}^{(l)} \end{bmatrix}}_{N\text{ times}}$$

$$\bold{Z}^{(l)}=\Big(\bold{W}^{(l)}\Big)^T\bold{A}^{(l-1)} + \bold{B}^{(l)}$$

At output layer ($l=L$):

$$\bold{E}^{(L)}=\dfrac{\partial \mathcal{L}}{\partial \bold{A}^{(L)}} \odot \bold{f}^{\prime}\Big( \bold{Z}^{(L)} \Big)$$


$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(L)}}=\bold{A}^{(L-1)} \Big(\bold{E}^{(L)}\Big)^T=\begin{bmatrix} \bold{a}_1^{(L-1)} \vert \bold{a}_2^{(L-1)} \vert \dots \vert \bold{a}_n^{(L-1)} \vert \dots \vert \bold{a}_N^{(L-1)} \end{bmatrix}\begin{bmatrix} \underline{\Big(\bold{e}_1^{(L)}\Big)^T} \\ \underline{\Big(\bold{e}_2^{(L)}\Big)^T} \\ \vdots \\ \underline{\Big(\bold{e}_n^{(L)}\Big)^T} \\ \vdots \\ \Big(\bold{e}_N^{(L)}\Big)^T \end{bmatrix}=\sum_{n=1}^{N} \bold{a}_n^{(L-1)}\Big(\bold{e}_n^{(L)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(L)}}=\bold{E}^{(L)}\begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}=\sum_{n=1}^{N}\bold{e}_n^{(L)}$$


At l-th layer ($1 \leq l < L$):

$$\bold{E}^{(l)}=\bold{W}^{(l+1)}\bold{E}^{(l+1)}\odot \bold{f}^{\prime}\Big( \bold{Z}^{(l)} \Big)$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{W}^{(l)}}=\bold{A}^{(l-1)}\Big(\bold{E}^{(l)}\Big)^T$$

$$\dfrac{\partial \mathcal{L}}{\partial \bold{b}^{(l)}}=\bold{E}^{(l)}\begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}=\sum_{n=1}^{N}\bold{e}_n^{(l)}$$