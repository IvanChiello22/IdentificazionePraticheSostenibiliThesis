# 42) JAX/Flax — T19 NEGATIVO: update loop senza checkpoint
import jax, jax.numpy as jnp
key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (10,2)); b = jnp.zeros((2,))
def step(w,b,x,y,lr=1e-2):
    logits = x @ w + b
    probs = jax.nn.softmax(logits)
    loss = -jnp.mean(jnp.log(probs[jnp.arange(y.size), y]))
    grad_w = (x.T @ (probs - jax.nn.one_hot(y,2))) / x.shape[0]
    grad_b = jnp.mean(probs - jax.nn.one_hot(y,2), axis=0)
    return w - lr*grad_w, b - lr*grad_b
for e in range(3):
    x = jax.random.normal(key, (32,10)); y = jax.random.randint(key, (32,), 0, 2)
    w,b = step(w,b,x,y)
print('done')
