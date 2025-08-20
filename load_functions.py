import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import weibull_min

    
    
# Enable double-precission floating-point to maximize speed
jax.config.update("jax_enable_x64", True)


# Smooth helper functions for numerical stability
def smooth_indicator(x, y, beta=50.0):
    return jax.nn.sigmoid(beta * (x - y)) + jax.nn.sigmoid(beta * (y - x))
    
def smooth_positive(x):
    return jax.nn.softplus(x)


def loglikelihood_jax(params, patients, state, obstime, deltaobstime, nstates, parscale):
        
    epsilon = 1e-12
    N2 = nstates * nstates
    vij_vec, sij_vec, aij_vec, bij_vec = jnp.split(params, [N2, 2*N2, 3*N2])

    # Smooth positivity
    vij = smooth_positive(vij_vec.reshape((nstates, nstates), order='F')) / parscale
    sij = smooth_positive(sij_vec.reshape((nstates, nstates), order='F')) / parscale

    # aij / bij constraints
    aij = aij_vec.reshape((nstates, nstates), order='F')
    aij = aij.at[jnp.arange(nstates), jnp.arange(nstates)].set(0.0)
    if nstates >= 2:
        aij = aij.at[nstates - 1, nstates - 2].set(-jnp.sum(aij[nstates - 1, 0:(nstates - 2)]))
    aij = aij.at[:, nstates - 1].set(-jnp.sum(aij[:, 0:(nstates - 1)], axis=1))
    aij = aij.at[jnp.arange(nstates), jnp.arange(nstates)].set(0.0)

    bij = bij_vec.reshape((nstates, nstates), order='F')
    bij = bij.at[jnp.arange(nstates), jnp.arange(nstates)].set(0.0)
    if nstates >= 2:
        bij = bij.at[nstates - 1, nstates - 2].set(1.0 - jnp.sum(bij[nstates - 1, 0:(nstates - 2)]))
    bij = bij.at[:, nstates - 1].set(1.0 - jnp.sum(bij[:, 0:(nstates - 1)], axis=1))
    
    # Flatten for indexing
    aij_flat = aij.flatten(order='F')
    bij_flat = bij.flatten(order='F')
    vij_flat = vij.flatten(order='F')
    sij_flat = sij.flatten(order='F')

    # Row-wise inputs
    patients_next = jnp.roll(patients, -1)
    state_next = jnp.roll(state, -1)
    same_patient = (patients == patients_next)
    valid_dt = ~jnp.isnan(deltaobstime)
    final_mask = same_patient & valid_dt

    state_next_eff = jnp.where(final_mask, state_next, state)
    dt_eff         = jnp.where(final_mask, deltaobstime, 1.0)
    changed_state_smooth = smooth_indicator(state, state_next_eff, beta=50.0)

    # Define necessary functions for likelihood calculation
    def jax_p(i, j, aij, bij, t, nstates):
        idx = j * nstates + i
        out = t * aij[idx] + bij[idx]
        p_out = jax.nn.sigmoid(out)
        return jnp.clip(p_out, a_min=epsilon, a_max=1.0 - epsilon)

    def jax_f(i, j, vij, sij, x, nstates):
        idx = j * nstates + i
        shape = jnp.clip(vij[idx], a_min=1e-6)
        scale = jnp.clip(sij[idx], a_min=1e-6)
        x_pos = jnp.clip(x, a_min=1e-10)
        x_scaled = x_pos / scale
        out = (shape / scale) * jnp.power(jnp.clip(x_scaled, a_min=1e-12), shape - 1.0) * jnp.exp(-jnp.power(jnp.clip(x_scaled, a_min=1e-12), shape))
        return jnp.clip(out, a_min=epsilon)

    def jax_F(i, j, vij, sij, x, nstates):
        idx = j * nstates + i
        shape = jnp.clip(vij[idx], a_min=1e-6)
        scale = jnp.clip(sij[idx], a_min=1e-6)
        x_pos = jnp.clip(x, a_min=1e-10)
        x_scaled = x_pos / scale
        out = 1.0 - jnp.exp(-jnp.power(jnp.clip(x_scaled, a_min=1e-12), shape))
        return jnp.clip(out, a_min=epsilon, a_max=1.0 - epsilon)

    def jax_H(i, t, x, aij, bij, vij, sij, nstates):
        js = jnp.arange(nstates)
        p_vals = vmap(jax_p, in_axes=(None, 0, None, None, None, None))(i, js, aij, bij, t, nstates)
        F_vals = vmap(jax_F, in_axes=(None, 0, None, None, None, None))(i, js, vij, sij, x, nstates)
        mask   = 1.0 - smooth_indicator(js, i, beta=50.0)
        out = jnp.sum(p_vals * F_vals * mask)
        return jnp.clip(out, a_min=epsilon, a_max=1.0 - epsilon)

    # Vectorized computation for likelihood
    log_p_vals = jnp.log(vmap(jax_p, in_axes=(0, 0, None, None, 0, None))(
        state, state_next_eff, aij_flat, bij_flat, obstime, nstates))

    log_f_vals = jnp.log(vmap(jax_f, in_axes=(0, 0, None, None, 0, None))(
        state, state_next_eff, vij_flat, sij_flat, dt_eff, nstates))

    H_vals = vmap(jax_H, in_axes=(0, 0, 0, None, None, None, None, None))(
        state, obstime, dt_eff, aij_flat, bij_flat, vij_flat, sij_flat, nstates)

    term_different_state = log_p_vals + log_f_vals
    term_same_state      = jnp.log1p(-H_vals)

    per_row = changed_state_smooth * term_different_state + (1.0 - changed_state_smooth) * term_same_state
    
    total_log_likelihood = jnp.sum(per_row * final_mask)

    return -total_log_likelihood

    
jit_loglikelihood_jax = jax.jit(loglikelihood_jax, static_argnames=['nstates', 'parscale'])

# Define wrapper functions to pass the arguments in the optimization
def fun_wrapper(params_np, *args):
    obstimes_df, nstates, parscale = args 
    
    # Convert numpy arrays from the DataFrame to JAX arrays
    patients = jnp.array(obstimes_df["PATIENT"].values, dtype=jnp.int64)
    state = jnp.array(obstimes_df["state"].values, dtype=jnp.int64)
    obstime = jnp.array(obstimes_df["obstime"].values, dtype=jnp.float64)
    deltaobstime = jnp.array(obstimes_df["deltaobstime"].values, dtype=jnp.float64)
    
    # Convert the parameters to a JAX array
    params_jax = jnp.array(params_np, dtype=jnp.float64)
    
    return np.asarray(jit_loglikelihood_jax(params_jax, patients, state, obstime, deltaobstime, nstates, parscale))

def jac_wrapper(params_np, *args):
    obstimes_df, nstates, parscale = args
    
    # Convert numpy arrays from the DataFrame to JAX arrays 
    patients = jnp.array(obstimes_df["PATIENT"].values, dtype=jnp.int64)
    state = jnp.array(obstimes_df["state"].values, dtype=jnp.int64)
    obstime = jnp.array(obstimes_df["obstime"].values, dtype=jnp.float64)
    deltaobstime = jnp.array(obstimes_df["deltaobstime"].values, dtype=jnp.float64)
    
    # Convert the parameters to a JAX array
    params_jax = jnp.array(params_np, dtype=jnp.float64)
    
    grad_fn = jax.grad(jit_loglikelihood_jax, argnums=0)
    gradient = grad_fn(params_jax, patients, state, obstime, deltaobstime, nstates, parscale)
    return np.asarray(gradient)


# Parameter statistics
def analyze_parameters(array_list):
    # Convert the list of arrays into a single 2D NumPy array
    data_matrix = np.stack(array_list, axis=0)

    # Number of observations (arrays) and degrees of freedom
    n = data_matrix.shape[0]
    df = n - 1

    # Initialize lists to store the results
    means = np.mean(data_matrix, axis=0)
    sds = np.std(data_matrix, axis=0)
    
    # Calculate standard error of the mean
    sems = sds / np.sqrt(n)
    
    # Calculate the t-critical value for a 95% CI
    t_critical = st.t.ppf(0.975, df)
    
    # Calculate the confidence interval bounds
    ci_lower = means - t_critical * sems
    ci_upper = means + t_critical * sems
    
    # Perform a one-sample t-test against a null hypothesis of mean = 0
    t_values, p_values = st.ttest_1samp(data_matrix, 0, axis=0)


    # Flatten all the results into 1-dimensional arrays before creating the DataFrame
    means = means.flatten()
    sds = sds.flatten()
    ci_lower = ci_lower.flatten()
    ci_upper = ci_upper.flatten()
    t_values = t_values.flatten()
    p_values = p_values.flatten()

    # Create a DataFrame to present the results
    results_df = pd.DataFrame({
        'Parameter': [f'Param {i+1}' for i in range(len(means))],
        'Mean': means,
        'SD': sds,
        'CI Lower (95%)': ci_lower,
        'CI Upper (95%)': ci_upper,
        't-value': t_values,
        'p-value': p_values
    })
    
    # Format the DataFrame for cleaner output
    results_df = results_df.round(4)

    return results_df