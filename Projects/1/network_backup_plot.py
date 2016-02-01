import matplotlib.pyplot as plt
import numpy


def day_to_number(str):
    map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0,
           'Sunday': 6.0}
    return map[str]


def number_from_end_string(str):
    return [float(s) for s in str.split('_') if s.isdigit()][-1]


file = numpy.genfromtxt('../../Datasets/network_backup_dataset.csv',
                        delimiter=',', skip_header=1,
                        converters={1: day_to_number, 3: number_from_end_string, 4: number_from_end_string})


plot_data = []
for i in range(20):
    plot_data.append([0.0] * 5)

for i in range(3538):
    row = file[i]
    plot_data[int(((row[0]-1)*7)+row[1])][int(row[3])] += row[5]

print [x[0] for x in plot_data]

p1 = plt.plot(range(20), [x[0] for x in plot_data])
p2 = plt.plot(range(20), [x[1] for x in plot_data], 'red')
p3 = plt.plot(range(20), [x[2] for x in plot_data], 'magenta')
p4 = plt.plot(range(20), [x[3] for x in plot_data], 'green')
p5 = plt.plot(range(20), [x[4] for x in plot_data], 'orange')
plt.xlabel('Days')
plt.ylabel('Size of file (GB)')
plt.title('Workflow wise network size.')
plt.grid(True)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('work_flow_0', 'work_flow_1', 'work_flow_2', 'work_flow_3', 'work_flow_4'))

plt.show()

