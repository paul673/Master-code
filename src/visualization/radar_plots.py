
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import sys
from pycirclize import Circos



def get_sorted_targed_scent_indicies(target):
    return np.argsort(-np.array(target)) 


def get_most_important_indicies(number,target, labels):
    #labels= openpom_keys
    indices=get_sorted_targed_scent_indicies(target)
    most_important_indicies = indices[:number]
    most_important_keys = []
    for i in most_important_indicies:
        most_important_keys.append(labels[i])
    return most_important_keys


def create_radar_plots_set(plot_df,plot_target_df, dirname,figsize=(2.165, 2.3), bbox=(0.5, -0.5),grid_label_formatter=None,label_kws_handler=None,grid_label_kws = None):
    

    for i, target_name in enumerate(plot_df.index):
        target_df = plot_df.loc[[target_name]]
        reward_val=target_df.index[0].split(":")[1]
        target_df.index = ["Molecule"]
        target_df = pd.concat([target_df, plot_target_df])
        # Initialize Circos instance for radar chart plot
        circos = Circos.radar_chart(
            target_df,
            vmax=1,
            marker_size=2,
            #cmap=dict(Hero="salmon", Warrior="skyblue", Wizard="lime", Assassin="magenta"),
            grid_interval_ratio=0.25,
            line_kws_handler=lambda _: dict(lw=1, ls="solid"),
            marker_kws_handler=lambda _: dict(marker="D", ec="grey", lw=0.5),
            grid_label_formatter=grid_label_formatter,
            label_kws_handler=label_kws_handler,
            grid_label_kws = grid_label_kws,
        )

        # Plot figure & set legend on upper right
        fig = circos.plotfig(figsize=figsize)
        #circos.ax.set_position([0.2, 0.2, 0.96, 0.96])
        circos.ax.set_title("")  # no title
        circos.ax.legend(loc="lower center",bbox_to_anchor=bbox, ncols=2)
        #plt.legend(False)
        plt.savefig(f"../../images/{dirname}/mols/radar{i}r{reward_val}.pdf", bbox_inches="tight", pad_inches=0.01)  # minimal padding)
        
    return fig

def get_radar_charts(df,savedir, vanillin_data, labels, number=6):
    important_cols = get_most_important_indicies(number,list(vanillin_data.iloc[0]), labels)
    rews = list(df["R(x)"])
    df = df[important_cols].set_index(pd.Series([f"{i+1} r:{rews[i]:.2f}" for i in range(len(df))]))
    t_data = vanillin_data[important_cols]
    create_radar_plots_set(
    df,
    t_data, 
    savedir)

if __name__ == "__main__":
    input_json_file = sys.argv[1]
    with open(input_json_file, 'r') as f:
        inputs = json.load(f)
    df = pd.read_csv(inputs["df_file"],index_col=0)
    vanillin_df = pd.read_csv(inputs["vanillin_df_file"],index_col=0)
    print(inputs)
    print(list(vanillin_df.iloc[0]))

    get_radar_charts(
        df, 
        inputs['savedir'],
        vanillin_df,
        inputs["labels"],
        inputs["number"],
        )