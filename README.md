# Hessian_Optimizer



This repository contains the code for a Deep Learning Second-order Optimizer. The idea behind our second-order optimizer is to enable efficient and systematic training of NNs as learning non-use-case specific models has been shown to be a challenging task in DL and even the choice of the optimizer has become a hyperparameter. We propose AdaPrHess, an optimizer that uses second-order information in the update rule. We calculate first and second momentum as follows:


$$ m_t =\frac{(1-\beta_{1})\sum_{i=1}^{t} \beta_1^{t-i}\textbf{{g}}_{i}}{(1-\beta_1^{t})} $$

 $$ v_t =\sqrt{\frac{(1-\beta_{2})\sum_{i=1}^{t} \beta_1^{t-i}\textbf{{d}}_i^{2}}{(1-\beta_2^{t})}} $$

The first momentum is as Adam, compute the momentum gradient. In the second momentum, the denominator $\textbf{{v}}_{t}$, we are using the diagonal of the Hessian instead of the gradient, referred to as $\textbf{{d}}_i^{2}$. This small difference has a powerful effect as we dynamically incorporated both kinds of knowledge in order to approximate the loss function: gradient and Hessian. Utilizing this information improves the efficiency of the optimization process and helps converge to a solution more rapidly as the second-order derivatives are used to determine the nature of a critical point, such as 
