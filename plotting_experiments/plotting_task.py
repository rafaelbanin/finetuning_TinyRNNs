from plotting import *
import numpy as np

if __name__ == '__main__':
    ### reversal_learning_task
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([80]) * 0.7
    # pr[50:] = 0.3
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(1-pr, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 80])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'reversal_learning_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    ### two_stage_task
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([300]) * 0.8
    # pr[25:50] = 0.2
    # pr[100:120] = 0.2
    # pr[160:180] = 0.2
    # pr[240:] = 0.2
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(1-pr, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 300])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'two_stage_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    ### novel_two_stage_task
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([300]) * 0.8
    # pr[40:100] = 0.4
    # pr[160:240] = 0.2
    # pr[240:] = 0.4
    # pr_1 = 1 - pr
    # pr_1[pr_1==0.6] = 0.4
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(pr_1, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 300])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'novel_two_stage_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    ### novel_two_stage_task_transition
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([300]) * 0.8
    # pr[40:] = 0.2
    # pr_1 = 1 - pr
    # plt.plot(pr, color='C1', linewidth=1)
    # plt.plot(pr_1, color='C4', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 300])
    # plt.xlabel('Trial')
    # plt.ylabel('Transition\nprob.')
    # plt.savefig(FIG_PATH / 'novel_two_stage_task_transition.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    ### gillan_two_stage_task_reward
    # fig, ax = plot_start(figsize=(1, 0.5))
    # fpath = r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\Gillandata\Experiment 1\twostep_data_study1\3A1COHJ8NJVPKFSRS1WC7SOUCMHH84.csv'
    # from datasets.GillanHumanDataset import load_subject_data_csv
    # task_dict = load_subject_data_csv(fpath)
    #
    # plt.plot(task_dict['drift_1'], color='C3', linewidth=0.8, alpha=0.8)
    # plt.plot(task_dict['drift_2'], color='C3', linewidth=0.8, alpha=0.8)
    # plt.plot(task_dict['drift_3'], color='C0', linewidth=0.8, alpha=0.8)
    # plt.plot(task_dict['drift_4'], color='C0', linewidth=0.8, alpha=0.8)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 200])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'gillan_two_stage_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    ### 4_arm_task
    # fig, ax = plot_start(figsize=(1, 0.5))
    # fpath = r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\4ArmBandit_DataAllSubjectsRewards.csv'
    # data = pd.read_csv(fpath)
    # data = data[data['id'] == 1]
    # plt.plot(data['reward_c1'], color='C3', linewidth=0.8, alpha=0.8)
    # plt.plot(data['reward_c2'], color='C0', linewidth=0.8, alpha=0.8)
    # plt.plot(data['reward_c3'], color='C2', linewidth=0.8, alpha=0.8)
    # plt.plot(data['reward_c4'], color='C8', linewidth=0.8, alpha=0.8)
    # plt.ylim([0, 100])
    # plt.yticks([0, 100])
    # plt.xticks([0, 150])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward')
    # plt.savefig(FIG_PATH / '4_arm_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    ### 3-arm task
    fig, ax = plot_start(figsize=(1, 0.5))
    r1 = np.array([5,1,9,5,1,9,1,5,8,2,8,4,2,8,2,4])/10
    r2 = np.array([9,5,5,1,9,5,5,9,2,4,4,2,8,4,4,8])/10
    r3 = np.array([1,9,1,9,5,1,9,1,4,8,2,8,4,2,8,2])/10
    # copy 10 times
    r1 = np.repeat(r1, 10)
    r2 = np.repeat(r2, 10)
    r3 = np.repeat(r3, 10)
    plt.plot(r1, color='C3', linewidth=0.8, alpha=0.8)
    plt.plot(r2, color='C0', linewidth=0.8, alpha=0.8)
    plt.plot(r3, color='C2', linewidth=0.8, alpha=0.8)
    plt.ylim([0, 1])
    plt.yticks([0, 1])
    plt.xticks([0, 160])
    plt.xlabel('Trial')
    plt.ylabel('Reward\nprob.')
    plt.savefig(FIG_PATH / '3_arm_task.pdf', dpi=300, bbox_inches='tight')
    plt.show()