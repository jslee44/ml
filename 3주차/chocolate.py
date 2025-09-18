import numpy as np
data = np.loadtxt('./sample_data/chocolate_rating.csv', delimiter=',')

high_level = data[data[:,3] >= 4]
high_id = high_level[:,0]
print('Number of high level chocotate:', high_id.size)

high_cacao = high_level[:,2]
unique_values, value_counts = np.unique(high_cacao, return_counts=True)
print('Cacao content:', unique_values)
print('Frequency by content level:', value_counts)

max_index = np.argmax(value_counts)
print('Most common cacao content (high-rated):', unique_values[max_index])