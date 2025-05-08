import pandas as pd
import numpy as np

# Sample DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

# Combine rows into a NumPy array
result = np.array([df1.iloc[i].values.tolist() + df2.iloc[i].values.tolist() for i in range(len(df1))])

print(result)


# Create two sample DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

# Convert DataFrames to NumPy arrays
array1 = df1.to_numpy()
array2 = df2.to_numpy()

# Combine rows into a 2D array
result_array = np.hstack((array1, array2))

print(result_array)



# Sample DataFrame
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'C', 'C'],
    'value1': [1, 2, 3, 4, 5, 6],
    'value2': [7, 8, 9, 10, 11, 12]
})

# Group by 'group' and create a 2D array for each group
grouped = df.groupby('group').agg(lambda x: list(x)).reset_index()

print(grouped[['value1', 'value2']].values)

# Convert to a 2D numpy array
result = np.array(grouped[['value1', 'value2']].values.tolist())



print(result)