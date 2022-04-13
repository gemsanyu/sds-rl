
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Read results
# none, t_1, t_5 = sim_none.to_dataframe(), sim_t1.to_dataframe(), sim_t5.to_dataframe()
# none['name'], t_1['name'], t_5['name'] = "None", "Timeout (1)", "Timeout (5)"
# benchmark = pd.concat([none, t_1, t_5], ignore_index=True)
# benchmark['mean_slowdown'] = np.log(benchmark['mean_slowdown'])
# benchmark['consumed_joules'] = np.log(benchmark['consumed_joules'])
# # Slowdown
# plt.figure(figsize=(12,4))
# plt.subplot(1, 2, 1)
# plt.plot('name', 'mean_slowdown', data=benchmark)
# plt.grid(axis='y')
# plt.ylabel("Averaged Slowdown (s)")

# # Energy consumed
# plt.subplot(1, 2, 2)
# plt.plot('name', 'consumed_joules', data=benchmark)
# plt.grid(axis='y')
# plt.ylabel("Energy Consumed (J)")

# # Show
# plt.show()