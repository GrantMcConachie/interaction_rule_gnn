"""
Script that generates all the graph data
"""

import os
import h5py
import yaml
import numpy as np

from dataset_utils import synthetic_systems, synthetic_utils, fish_utils, utils


def generate_fish_graphs(fp, save_fp):
    """
    Gets location data out of the raw files and saves them as pkl files
    """
    # list files
    fish_files = os.listdir(fp)

    # loop though files
    for f in fish_files:
        fish_h5 = h5py.File(os.path.join(fp, f))

        # take out location data
        fish_hdf = fish_h5['tracks']

        # generate state vectors
        state_vectors = fish_utils.generate_x_dot(fish_hdf)

        # generate graphs
        graphs = utils.generate_graphs(state_vectors)

        # saving
        f = f.replace('.h5', '.pkl')
        utils.save_graphs(os.path.join(save_fp, f), graphs)


def generate_spring_mass_graphs(save_fp, config, dynamic, make_gif, seed):
    """
    Docstring for generate_spring_mass
    
    save_fp (str) - Filepath to save spring mass system data to
    config (str) - filepath to config file
    dynamic (bool) - Have the spring system be dynamic or not
    """
    # load config
    config = yaml.safe_load(open(config, 'r'))

    # generate 5 different initializations
    for i in range(5):
        rng = np.random.default_rng([seed, i])
        
        # dynamic edges of static edges
        if dynamic:
            sm = synthetic_systems.springMassDynamicEdges(
                fov_angle=config["fov_angle"],
                n_masses=config["n_masses"],
                arena_size=config["arena_size"],
                force_field_proportion=config["force_field_proportion"],
                spring_constant=config["spring_constant"],
                damping_coef=config["damping_coef"],
                init_vel_norm=config["init_vel_norm"],
                time=config["time"],
                dt=config["dt"],
                rng=rng
            )
            pos, vel, edges = sm.simulate()
        
        else:
            sm = synthetic_systems.springMassSystem(
                n_masses=config["n_masses"],
                arena_size=config["arena_size"],
                force_field_proportion=config["force_field_proportion"],
                spring_constant=config["spring_constant"],
                damping_coef=config["damping_coef"],
                init_vel_norm=config["init_vel_norm"],
                time=config["time"],
                dt=config["dt"],
                rng=rng
            )
            pos, vel, edges = sm.simulate()
        
        # saving animations
        if make_gif:
            synthetic_utils.animate_system(
                pos,
                edges,
                save_path=os.path.join(save_fp, 'gifs', f'trial_{i}.gif'),
                force_field_radius=sm.force_field_size
            )

        # save graphs
        state_vars = synthetic_utils.make_state_vars(pos, vel, edges=edges, dt=config['dt'])
        graphs = utils.generate_graphs(state_vars, gt_edges=edges)
        utils.save_graphs(os.path.join(save_fp, 'graphs', f'trial_{i}.pkl'), graphs)

    # save config
    with open(os.path.join(save_fp, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    # print('Creating fish graphs')

    # # 8 fish
    # generate_fish_graphs(
    #     fp='data/fish/raw data/8fish',
    #     save_fp='data/fish/processed/8fish'
    # )

    # # 10 fish
    # generate_fish_graphs(
    #     fp='data/fish/raw data/10fish/DATA',
    #     save_fp='data/fish/processed/10fish'
    # )

    print('Creating spring mass system graphs')

    # spring mass
    generate_spring_mass_graphs(
        save_fp='data/spring_mass/static_graph',
        config='configs/spring_mass_static_graph.yaml',
        dynamic=False,
        make_gif=False,
        seed=12345
    )

    # dynamic spring mass
    generate_spring_mass_graphs(
        save_fp='data/spring_mass/dynamic_graph',
        config='configs/spring_mass_dynamic_graph.yaml',
        dynamic=True,
        make_gif=False,
        seed=12345
    )
