I'm genuinely astounded that snakes (active contours) can slowly work their way in from a standard ellipse to a tight yet still smooth curve around an object,
embracing its curvature while still not overfitting. I'd love to dive a little deeper into the math behind this and see how the snake updates its parameters
at each step to adjust the curve (I imagine through standard gradient descent, but I'd still love to learn more), especially seeing each step of the snake's convergence.
This exercise has made my next steps for my climbing hold classifier far clearer as I can annotate bounding boxes around my holds by myself, but the snake will perform
the exact polygon segmentation on its own, saving me tons of time (ex. labeling a polygon takes around 40-60 seconds for me, but the snake only takes one minute without
my active effort).
I will absolutely be using snakes for at least one form of segmentation, but I do still think that I'll need others to determine the segmentation of an object
based on primarily its texture (since climbing holds stand out from their backgrounds predominantly by their textures).
I do think I understand how snakes work, including the math behind them, far more than I did in class, especially in terms of the alpha/beta parameters as well as
why we blur before running a snake in the first place. The balance between smoothness and accuracy is truly fascinating to me.