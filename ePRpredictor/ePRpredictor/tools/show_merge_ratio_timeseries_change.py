import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# ratio_of_merged_pr_in_this_proj_before_pr
flutter = [0.023408977, 0.034769095, 0.07630667, 0.18536751, 0.22107957, 0.24996798, 0.25323159, 0.24871165, 0.24638431, 0.26760176]

nodejs = [0.02870255, 0.038829602, 0.036632065, 0.03984251, 0.03495248, 0.04666718, 0.04720102, 0.05799388, 0.057347696, 0.070496425]

tensorflow = [0.246726, 0.23587577, 0.12083596, 0.080871336, 0.12298836, 0.14631586, 0.12619422, 0.10931017, 0.15726198, 0.19979835]

angular = [0.10493055, 0.06425685, 0.05432724, 0.063695915, 0.05389444, 0.048380993, 0.043902475, 0.041540537, 0.040745173, 0.039025463]

mp_list = {'flutter/flutter':flutter, \
            'nodejs/node': nodejs, \
            'tensorflow/models': tensorflow, \
            'angular/angular' : angular
            }

def output_importance_plt(mp_list, dir):
    x_ticks = [i+1 for i in range(10)]
    x = np.arange(len(x_ticks))

    plt.figure(figsize=(25, 15))
    texts = []
    for feature in mp_list:

        plt.plot(x, mp_list[feature], label=feature, linewidth=5.0)
        for a, b in zip(x, mp_list[feature]):
            texts.append(
                plt.text(a, b, '%.3f'%b, ha='center', va= 'bottom', fontsize=40)
            )
    
    plt.xticks([r for r in x], x_ticks, fontsize=30, rotation=50)
    plt.yticks(fontsize=30)

    plt.xlabel(u'Time ID', fontsize=50)
    plt.ylabel(u'Importance', fontsize=50)

    # plt.title(u'Title', fontsize=10)

    plt.legend(fontsize=30, prop={'style':'oblique','weight':'bold', 'size':27})
    adjust_text(texts, only_move={'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.8))
    
    plt.subplots_adjust(left=0.07, right=0.97, top=0.98, bottom=0.1)

    # plt.savefig(dir, bbox_inches='tight')
    plt.savefig(dir)

output_importance_plt(mp_list, "./test2.pdf")