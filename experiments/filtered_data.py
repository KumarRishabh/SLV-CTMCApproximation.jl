import pandas as pd

# Load the option prices data
file_path = "NVDA_options_data.csv"  # Update with your file path
options_data = pd.read_csv(file_path)

# Define common strike prices
common_strike_prices = [50, 75, 100, 125, 150, 175, 200]

# Filter data to include only rows with the common strike prices
filtered_data = options_data[options_data["strike"].isin(common_strike_prices)]

# Pivot the data to have expiry dates as rows and strike prices as columns
pivot_data = filtered_data.pivot(index="expirationDate", columns="strike", values="impliedVolatility")

# Save the filtered data to a CSV for use in Julia
filtered_file_path = "NVDA_filtered_options_expiry.csv"
pivot_data.to_csv(filtered_file_path)

# Display the first few rows of the filtered data
print("Filtered Data (Pivoted):")
print(pivot_data.head())

