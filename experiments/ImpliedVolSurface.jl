using CSV
using DataFrames
using Plots

# Load the CSV data
options_data = CSV.read("NVDA_options_data.csv", DataFrame)

# Extract relevant columns
strike_prices = options_data.strike
time_to_expiry = options_data.timeToExpiry
implied_vols = options_data.impliedVolatility

# Convert the data into a grid for plotting

# Load the filtered CSV data
filtered_options_data = CSV.read("NVDA_filtered_options_expiry.csv", DataFrame)

# Extract relevant columns from the filtered data
# strike_prices = filtered_options_data[:, Not(:expirationDate)]
strike_prices = parse.(Float64, names(filtered_options_data)[3:end])
# time_to_expiry = filtered_options_data.timeToExpiry
expiry_date = filtered_options_data.expirationDate[:]

x = unique(strike_prices)
y = unique(expiry_date)
z = zeros(length(y), length(x))

for (i, date) in enumerate(expiry_date)
    row = filtered_options_data[filtered_options_data.expirationDate .== date, :]
    for (j, strike) in enumerate(strike_prices)
        val = row[1, string(strike)]
        z[i,j] = ismissing(val) ? NaN : val
    end
end

z

using Plots
pyplot()
tight_layout()
  # Use the PyPlot backend for more control over label positions
date_nums = Float64[(d - minimum(expiry_date)).value for d in expiry_date]
y_midpoint = (maximum(date_nums) + minimum(date_nums)) / 2

# Plot surface
p = surface(
    strike_prices, expiry_date, z,
    xlabel = "Strike Price",
    ylabel = "",
    zlabel = "Implied Volatility",
    title = "Implied Volatility Surface for NVDA Options",
    legend = true,
    c = :viridis,
    size = (800, 600)
)
x_pos = -1.5  # Adjust based on your axis scale
y_pos = maximum(expiry_date) / 2  # Center along the expiration dates
annotate!(x_pos, y_pos, text("Expiration Date", 14, :left, :italic, rotation=90))
annotate!(-15, y_midpoint, text("Expiration Date", 12, :right, :bold))
# Adjust axis ticks
xticks!(p, 75:25:200)
yticks!(p, date_nums, string.(expiry_date))
# Customize ticks for better readability
# xticks!(p, 75:25:200)  # Adjust strike price ticks
# yticks!(p, 1:1:length(expiry_date))  # Fewer, evenly spaced expiration ticks

# Save or display the plot
savefig(p, "implied_volatility_surface_NVDA.svg")
display(p)