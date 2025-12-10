import pandas as pd

df = pd.read_csv(
    #"..\Datasets\Smart meters in London\halfhourly_dataset\halfhourly_dataset/block_110.csv",
    "..\Datasets\Micro PMU October 1 Dataset\_LBNL_a6_bus1_2015-10-01.csv",
    na_values=0
)
rows, columns = df.shape
print(f"{rows} Rows and {columns} Columns")