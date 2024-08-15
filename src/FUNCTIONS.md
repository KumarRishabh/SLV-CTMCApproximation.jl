# SLVCTMCApproximation Functions

## `calculateSufficientStats`

**Description**: 
This function calculates the sufficient statistics (mean and standard deviation) for the volatility process in the Heston model. It uses the parameters of the Cox-Ingersoll-Ross (CIR) process to compute these statistics over a given time horizon.

**Inputs**:
- `ν`: Long-term mean of the volatility process.
- `ϱ`: Rate of mean reversion.
- `κ`: Volatility of volatility.
- `v0`: Initial volatility.
- `T`: Time horizon.

**Outputs**:
- `mean`: The mean of the volatility process over the time horizon `T`.
- `std_dev`: The standard deviation of the volatility process over the time horizon `T`.

**Example**:
```julia
mean, std_dev = calculateSufficientStats(ν, ϱ, κ, v0, T)
```

## `VolatilityBins`

**Description**
The `VolatilityBins` function generates bins for the volatility process based on the calculated sufficient statistics. It allows for different binning modes and parameters to customize the binning process.

**Inputs**
- `ν`: Long-term mean of the volatility process.
- `ϱ`: Rate of mean reversion.
- `κ`: Volatility of volatility.
- `v0`: Initial volatility.
- `T`: Time horizon.
- `binning_mode` (optional): Mode of binning (default is "uniform").
- `epsilon` (optional): Small value to avoid division by zero (default is 0.0001).
- `γ` (optional): Parameter for binning (default is 2).
- `num_bins` (optional): Number of bins (default is 100).

**Outputs**
- Prints the mean and standard deviation of the volatility process.
- Generates and returns the bins for the volatility process.

**Example Usage**

```julia
# Define parameters for the volatility process
ν = 0.04
ϱ = 0.5
κ = 0.3
v0 = 0.02
T = 1.0

# Generate volatility bins
bins = VolatilityBins(ν, ϱ, κ, v0, T; binning_mode = "uniform", epsilon = 0.0001, γ = 2, num_bins = 100)
```


### `simulateQtransitions(Q, bins, T; v0 = 0.04)`

Simulates the transitions for the volatility process until time `T`.

#### Parameters:
- `Q`: Generator matrix for the volatility process.
- `bins`: Bins for the volatility process.
- `T`: Total time for the simulation.
- `v0`: Initial volatility (default is 0.04).

#### Returns:
- `state_transitions`: Array of state transitions.
- `transition_times`: Array of transition times.

### `simulatePriceProcess(transition_times, volatilitychain, μ, ν, ρ, κ, ϱ, S0, v0)`

Simulates the price process using the explicit price formula.

#### Parameters:
- `transition_times`: Array of transition times.
- `volatilitychain`: Array of volatility states.
- `μ`: Drift term.
- `ν`: Volatility of volatility.
- `ρ`: Correlation between the two Brownian motions.
- `κ`: Mean reversion rate.
- `ϱ`: Long-term mean of the volatility.
- `S0`: Initial price.
- `v0`: Initial volatility.

#### Returns:
- `log_prices`: Array of log prices.

### `multiple_price_volatility_simulations(T, μ, ν, ρ, κ, ϱ, S0, v0; num_simulations = 100)`

Performs multiple simulations of the price and volatility processes.

#### Parameters:
- `T`: Total time for the simulation.
- `μ`: Drift term.
- `ν`: Volatility of volatility.
- `ρ`: Correlation between the two Brownian motions.
- `κ`: Mean reversion rate.
- `ϱ`: Long-term mean of the volatility.
- `S0`: Initial price.
- `v0`: Initial volatility.
- `num_simulations`: Number of simulations to perform (default is 100).

#### Returns:
- `price_processes`: Array of price processes.
- `volatility_processes`: Array of volatility processes.
- `transition_times_processes`: Array of transition times for each simulation.