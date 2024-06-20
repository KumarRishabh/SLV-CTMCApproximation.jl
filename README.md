# Stochastic Volatility Options Pricing with Markov Chain Approximation

This Julia library implements the Markov chain approximation approach to simulate and price options using Stochastic Volatility models. This approach is based on the work of Zhenyu Cui, which provides an efficient method for dealing with stochastic processes in the context of financial modeling.

## Overview

Stochastic Volatility models are widely used in financial mathematics to capture the dynamic behavior of volatility in asset prices. These models help in better understanding and pricing of derivative products such as options. The Markov chain approximation method provides a numerical approach to approximate the continuous stochastic processes with a discrete Markov chain, facilitating easier and faster computation.

## Features

- Implementation of Markov chain approximation for various Stochastic Volatility models.
- Simulation of asset price paths under the chosen models.
- Pricing of European and American options.
- Support for common Stochastic Volatility models like Heston, SABR, and others.
- Flexible framework to add new models and pricing methods.

## Installation

To install this library, you can clone the repository and use Julia's package manager to include it in your project.

```sh
git clone https://github.com/yourusername/stochastic-volatility-options-pricing.git
cd stochastic-volatility-options-pricing
julia -e 'using Pkg; Pkg.add("path_to_your_package")'

<!-- Add Zhenyu Cui's references -->
