import pandas as pd

df = pd.read_csv("Datasets/Household_Electric_Power_Consumption/household_power_consumption.", sep=";", na_values='?')
df.dropna(inplace=True)

features = df[['Global_active_power',
                'Global_reactive_power',
                'Voltage',
                'Global_intensity',
                'Sub_metering_1',
                'Sub_metering_2',
                'Sub_metering_3']]
print(df.dtypes)
# df.to_csv('household_power_consumption.csv', index=False)