import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter


def get_no_cbf_data():
    folder_abs_path = os.path.abspath(os.path.join('models'))
    model_path = folder_abs_path + '\\' + 'no_cbf.xlsx'
    # print(model_path)

    df = pd.read_excel(model_path, header=0)
    data = df.to_numpy()
    # print(data)

    collision_rate = data[:, 0]
    overtake_rate = data[:, 1]

    return collision_rate, overtake_rate


def get_cbf_data():
    folder_abs_path = os.path.abspath(os.path.join('models'))
    collision_matrix = None
    overtake_matrix = None

    for i in range(10):
        step_num = (i + 1) * 10
        model_path = folder_abs_path + '\\' + 'cbf_' + str(step_num) + '_steps.xlsx'
        # print(model_path)

        df = pd.read_excel(model_path, header=0)
        data = df.to_numpy()
        # print(data)

        if i == 0:
            collision_matrix = data[:, 0]
            overtake_matrix = data[:, 1]
        else:
            collision_matrix = np.vstack((collision_matrix, data[:, 0]))
            overtake_matrix = np.vstack((overtake_matrix, data[:, 1]))

    return collision_matrix, overtake_matrix


def different_steps_line_chart():
    no_cbf_collision_rate, no_cbf_overtake_rate = get_no_cbf_data()
    # print(no_cbf_collision_rate)
    # print(no_cbf_overtake_rate)

    cbf_collision_rate, cbf_overtake_rate = get_cbf_data()
    print(cbf_collision_rate)
    print(cbf_overtake_rate)

    x = np.arange(0, 1000, 100) + 100  # iteration times
    # print(x)

    # collision plot
    plt.plot(x, no_cbf_collision_rate, label='No CBF')
    for i in range(10):
        label = str((i + 1) * 10) + ' steps'
        plt.plot(x, cbf_collision_rate[i], label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Collision Rate')
    plt.title('Collision Comparison')
    plt.legend(loc=3)  # lower left
    plt.show()

    # overtake plot
    plt.plot(x, no_cbf_overtake_rate, label='No CBF')
    for i in range(10):
        label = str((i + 1) * 10) + ' steps'
        plt.plot(x, cbf_overtake_rate[i], label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Overtake Rate')
    plt.title('Overtake Comparison')
    plt.legend(loc=2)  # upper left
    plt.show()


def different_steps_scatter_plot():
    no_cbf_collision_rate, no_cbf_overtake_rate = get_no_cbf_data()
    plt.scatter(no_cbf_collision_rate, no_cbf_overtake_rate, label='No CBF', c='#000000')

    cbf_collision_rate, cbf_overtake_rate = get_cbf_data()
    for i in range(10):
        label = str((i + 1) * 10) + ' steps'
        plt.scatter(cbf_collision_rate[i, :], cbf_overtake_rate[i, :], label=label)

    plt.xlabel('Collision Rate')
    plt.ylabel('Overtake Rate')
    plt.title('Scatter Plot')
    plt.legend(loc=1)  # upper right
    plt.show()


# using removal rate as labels
def different_steps_scatter_plot_new():
    max_arr = []
    min_arr = []

    # for i in range(1):
    for i in range(10):
        step_num = (i + 1) * 10
        folder_abs_path = os.path.abspath(os.path.join('models', 'step_removal', '0.5bad_behavior_cbf_filter',
                                                       'log data'))
        model_path = folder_abs_path + '\\' + str(step_num) + '_steps_removed.csv'
        # print(model_path)

        df = pd.read_csv(model_path, header=0)
        data = df.to_numpy()
        # print(data)

        removal_rates = []
        for j in range(10):
            iter_num = (j + 1) * 100
            if iter_num == 1000:
                iter_num = 990
            total_step_before_removal = np.sum(data[:iter_num, 1])
            num_of_steps_removed = np.sum(data[:iter_num, 3])
            # print(total_step_before_removal, num_of_steps_removed)
            rm_rate = num_of_steps_removed / total_step_before_removal
            removal_rates.append(rm_rate)
        # print(removal_rates)

        max_arr.append(np.amax(removal_rates))
        min_arr.append(np.amin(removal_rates))

    print(max_arr)
    print(min_arr)

    no_cbf_collision_rate, no_cbf_overtake_rate = get_no_cbf_data()
    plt.scatter(no_cbf_collision_rate, no_cbf_overtake_rate, label='No CBF', c='#000000')

    cbf_collision_rate, cbf_overtake_rate = get_cbf_data()
    print(cbf_collision_rate)
    for i in range(10):
        label = 'removal rate: ' + str(round(min_arr[i] * 100, 2)) + '% ~ ' + str(round(max_arr[i] * 100, 2)) + '%'
        plt.scatter(cbf_collision_rate[i, :], cbf_overtake_rate[i, :], label=label)

    plt.xlabel('Collision Rate')
    plt.ylabel('Overtake Rate')
    plt.title('Scatter Plot')
    plt.legend(loc=1)  # upper right
    plt.show()


# ----------------------------------------


def get_undesired_overtake_behavior_prob_data():
    folder_abs_path = os.path.abspath(os.path.join('models'))
    model_path = folder_abs_path + '\\' + '70_removed_steps_cbf_filter_iter_1000.xlsx'
    # print(model_path)

    df = pd.read_excel(model_path, header=0)
    data = df.to_numpy()
    # print(data)

    collision_rate = data[:, 0]
    overtake_rate = data[:, 1]

    return collision_rate, overtake_rate


def get_undesired_overtake_behavior_prob_data_no_cbf():
    folder_abs_path = os.path.abspath(os.path.join('models'))
    model_path = folder_abs_path + '\\' + 'no_cbf_filter_iter_1000.xlsx'
    # print(model_path)

    df = pd.read_excel(model_path, header=0)
    data = df.to_numpy()
    # print(data)

    collision_rate = data[:, 0]
    overtake_rate = data[:, 1]

    return collision_rate, overtake_rate


def different_prob_bar_plot():
    width = 0.4  # bar width

    # 0.0 ~ 1.0
    label = []
    for j in range(10 + 1):
        overtake_prob = round(j * 0.1, 1)
        label.append(str(overtake_prob))

    cbf_collision_rate, cbf_overtake_rate = get_undesired_overtake_behavior_prob_data()
    # print(collision_rate)
    # print(overtake_rate)
    no_cbf_collision_rate, no_cbf_overtake_rate = get_undesired_overtake_behavior_prob_data_no_cbf()

    plt.bar(np.arange(len(label)), no_cbf_collision_rate, width, label='no CBF')
    plt.bar(np.arange(len(label)) + width, cbf_collision_rate, width, label='with CBF')
    plt.xticks(np.arange(len(label)) + width / 2, label)
    plt.xlabel('Undesired Collision Behavior Probability')
    plt.ylabel('Rate')
    plt.title('Collision Rates among Different Probabilities')
    plt.legend(loc=2)  # upper left
    plt.show()

    plt.bar(np.arange(len(label)), no_cbf_overtake_rate, width, label='no CBF')
    plt.bar(np.arange(len(label)) + width, cbf_overtake_rate, width, label='with CBF')
    plt.xticks(np.arange(len(label)) + width / 2, label)
    plt.xlabel('Undesired Overtake Behavior Probability')
    plt.ylabel('Rate')
    plt.title('Overtake Rates among Different Probabilities')
    plt.legend(loc=2)  # upper left
    plt.show()


def write_undesired_overtake_behavior_prob_table():
    # x, prob, 0.0 ~ 1.0
    x = []
    for j in range(10 + 1):
        overtake_prob = str(round(j * 0.1, 1))
        x.append(overtake_prob)
    # print(x)

    rm_rates = []
    # loop 11 times - read & calculate
    for j in range(10 + 1):
    # for j in range(1):
        overtake_prob = str(round(j * 0.1, 1))
        # print(overtake_prob)
        csv_abs_path = os.path.abspath(os.path.join('models', 'undesired_overtake_behavior_prob', overtake_prob,
                                                    'data_log.csv'))
        # print(pkl_abs_path)

        df = pd.read_csv(csv_abs_path, header=0)
        data = df.to_numpy()
        # print(data)

        total_step_before_removal = np.sum(data[:, 1])
        num_of_steps_removed = np.sum(data[:, 3])
        # print(total_step_before_removal, num_of_steps_removed)
        rm_rate = num_of_steps_removed / total_step_before_removal
        # print(rm_rate)
        rm_rates.append(rm_rate)
    # print(rm_rates)

    data = np.vstack((x, rm_rates))
    print(data)

    # write
    folder_abs_path = os.path.abspath(os.path.join('models'))
    workbook = xlsxwriter.Workbook(folder_abs_path + '\\' + 'removal_rates_table.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_column('A1', ['behavior%', 'removal%'])

    col = 0
    for row, result in enumerate(data):  # iterating through content list
        worksheet.write_row(row, col + 1, result)  # 10 results = data

    workbook.close()


# ----------------------------------------


def main():
    # different_steps_line_chart()
    # different_steps_scatter_plot()

    # different_prob_bar_plot()
    # write_undesired_overtake_behavior_prob_table()

    different_steps_scatter_plot_new()


if __name__ == '__main__':
    main()
