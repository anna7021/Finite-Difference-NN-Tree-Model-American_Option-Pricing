import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

class FiniteDifferenceOptionPricing:
    def __init__(self, option_type, S, K, T, sigma, r, q, M, N):
        self.option_type = option_type  # "C" for call, "P" for put
        self.S = S    # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity
        self.sigma = sigma  # Volatility
        self.r = r    # Risk-free rate
        self.q = q    # Cost of carry
        self.M = M    # Number of price steps
        self.N = N    # Number of time steps

    def _generate_grid_params(self, price_multiplier):
        ds = self.S / self.M
        self.M = int(self.K / ds) * price_multiplier
        self.S_idx = int(self.S / ds)
        dt = self.T / self.N
        return ds, dt

    def explicit_fd(self):
        ds, dt = self._generate_grid_params(price_multiplier=4)
        discount_factor = 1 / (1 + self.r * dt)

        print(f"Generated grid: M = {self.M} price points, N = {self.N} time points")

        V_grid = np.zeros((self.M + 1, self.N + 1))
        S_array = np.linspace(0, self.M * ds, self.M + 1)
        T2M_array = self.T - np.linspace(0, self.N * dt, self.N + 1)

        # Boundary conditions
        if self.option_type == "call":
            V_grid[:, -1] = np.maximum(S_array - self.K, 0)
            V_grid[-1, :] = np.exp(-self.r * T2M_array) * (S_array[-1] * np.exp((self.r - self.q) * T2M_array) - self.K)
        else:
            V_grid[:, -1] = np.maximum(self.K - S_array, 0)
            V_grid[0, :] = np.exp(-self.r * T2M_array) * self.K

        # Coefficient functions
        aj = lambda j: 0.5 * (self.sigma**2 * j**2 - (self.r - self.q) * j) * dt
        bj = lambda j: 1 - self.sigma**2 * j**2 * dt
        cj = lambda j: 0.5 * (self.sigma**2 * j**2 + (self.r - self.q) * j) * dt

        coef_matrix = self._generate_tridiagonal_matrix(self.M, aj, bj, cj)

        for i in range(self.N - 1, -1, -1):
            Z = np.zeros(self.M - 1)
            Z[0] = aj(1) * V_grid[0, i + 1]
            Z[-1] = cj(self.M - 1) * V_grid[-1, i + 1]

            V_grid[1:self.M, i] = discount_factor * (coef_matrix @ V_grid[1:self.M, i + 1] + Z)

            # Early exercise for American options
            if self.option_type == "call":
                V_grid[1:self.M, i] = np.maximum(S_array[1:self.M] - self.K, V_grid[1:self.M, i])
            else:
                V_grid[1:self.M, i] = np.maximum(self.K - S_array[1:self.M], V_grid[1:self.M, i])

        return V_grid[self.S_idx, 0], S_array, V_grid[:, 0]

    def implicit_fd(self):
        ds, dt = self._generate_grid_params(price_multiplier=2)
        print(f"Generated grid: M = {self.M} price points, N = {self.N} time points")
        V_grid = np.zeros((self.M + 1, self.N + 1))
        S_array = np.linspace(0, self.M * ds, self.M + 1)
        T2M_array = self.T - np.linspace(0, self.N * dt, self.N + 1)
        # Boundary conditions
        if self.option_type == "call":
            V_grid[:, -1] = np.maximum(S_array - self.K, 0)
            V_grid[-1, :] = np.exp(-self.r * T2M_array) * (S_array[-1] * np.exp((self.r - self.q) * T2M_array) - self.K)
        else:
            V_grid[:, -1] = np.maximum(self.K - S_array, 0)
            V_grid[0, :] = np.exp(-self.r * T2M_array) * self.K

        # Coefficient functions
        aj = lambda j: 0.5 * j * ((self.r - self.q) - self.sigma**2 * j) * dt
        bj = lambda j: 1 + (self.r + self.sigma**2 * j**2) * dt
        cj = lambda j: 0.5 * j * (-(self.r - self.q) - self.sigma**2 * j) * dt

        coef_matrix = self._generate_tridiagonal_matrix(self.M, aj, bj, cj)
        M_inverse = np.linalg.inv(coef_matrix)

        for i in range(self.N - 1, -1, -1):
            Z = np.zeros(self.M - 1)
            Z[0] = aj(1) * V_grid[0, i]
            Z[-1] = cj(self.M - 1) * V_grid[-1, i]

            V_grid[1:self.M, i] = M_inverse @ (V_grid[1:self.M, i + 1] - Z)

            # Early exercise for American options
            if self.option_type == "call":
                V_grid[1:self.M, i] = np.maximum(S_array[1:self.M] - self.K, V_grid[1:self.M, i])
            else:
                V_grid[1:self.M, i] = np.maximum(self.K - S_array[1:self.M], V_grid[1:self.M, i])

        return V_grid[self.S_idx, 0], S_array, V_grid[:, 0]

    def _generate_tridiagonal_matrix(self, M, aj, bj, cj):
        diag_main = np.array([bj(j) for j in range(1, M)])
        diag_lower = np.array([aj(j) for j in range(2, M)])
        diag_upper = np.array([cj(j) for j in range(1, M - 1)])

        coef_matrix = np.diag(diag_main) + np.diag(diag_lower, -1) + np.diag(diag_upper, 1)
        return coef_matrix

# Example usage
pricing_model = FiniteDifferenceOptionPricing(option_type="P", S=36, K=40, T=0.5, sigma=0.4, r=0.06, q=0.01, M=125, N=50000)
explicit_price, S_explicit, V_explicit = pricing_model.explicit_fd()
implicit_price, S_implicit, V_implicit = pricing_model.implicit_fd()

print(f"Explicit Price: {explicit_price}")
print(f"Implicit Price: {implicit_price}")

# Plot
plt.plot(S_implicit, V_implicit, label="Implicit Method")
plt.plot(S_explicit, V_explicit, label="Explicit Method", linestyle='dashed')
plt.xlabel("Stock Price")
plt.ylabel("Option Value")
plt.title(f"American {pricing_model.option_type.capitalize()} Option Pricing - Finite Difference Methods")
plt.legend()
plt.grid()
plt.show()
plt.savefig("FD_American_opt.png")