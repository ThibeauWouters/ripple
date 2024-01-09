from ripple import lambdas_to_lambda_tildes, lambda_tildes_to_lambdas, Mc_eta_to_ms, ms_to_Mc_eta
import jax.numpy as jnp
import numpy as np

def test_conversion(N: int = 1, m_l : float = 0.5, m_u : float = 3.0, lambda_l : float = 0, lambda_u : float = 5000):
    
    # Generate random lambda and delta lambda tilde pairs
    N = 1
    og_lambda1 = np.random.uniform(low = lambda_l, high = lambda_l)
    og_lambda2 = np.random.uniform(low = lambda_l, high = lambda_l)
    
    # Generate random masses
    m1 = np.random.uniform(low = m_l, high = m_l)
    m2 = np.random.uniform(low = m_l, high = m_l)
    
    # Convert lambdas to lambda tildes
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([og_lambda1, og_lambda2, m1, m2]))
    # Convert back to lambdas
    lambda_1, lambda_2 = lambda_tildes_to_lambdas(jnp.array([lambda_tilde, delta_lambda_tilde, m1, m2]))
    
    lambda_1, lambda_2 = np.asarray(lambda_1), np.asarray(lambda_2)
    
    mse = np.mean((lambda_1 - og_lambda1)**2 + (lambda_2 - og_lambda2)**2)
    
    print(f"Mean squared error: {mse}")
    
    return None

def test_my_conversion():
    lambda_tilde, delta_lambda_tilde = 1480.88844601, 490.73150339
    mc, eta = 1.2063456, 0.13863835
    
    # Convert the masses
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    
    print(f"m1: {m1}, m2: {m2}")
    
    # Convert them
    lambda_1, lambda_2 = lambda_tildes_to_lambdas(jnp.array([lambda_tilde, delta_lambda_tilde, m1, m2]))
    
    print(f"lambda_1: {lambda_1}, lambda_2: {lambda_2}")
    

if __name__ == "__main__":
    test_my_conversion()