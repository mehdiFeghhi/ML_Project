import pickle
import matplotlib.pyplot as plt
with open('../checkpoint/Q_part2.pkl', 'rb') as f:
    list_of_q_part2 = pickle.load(f)

with open('../checkpoint/Q_Sarsa_part2.pkl', 'rb') as f:
    list_of_Sarsa_part2 = pickle.load(f)
x = [i for i in range(len(list_of_q_part2))]
fig, ax = plt.subplots()


plt.scatter(x, list_of_Sarsa_part2, color='r',  label='Sarsa')  #plot original function
plt.scatter(x, list_of_q_part2, color='g',  label='Q_learning') #plot calculated function
# plt.legend()
plt.show() #show the plot