#!/usr/bin/env python3

import os
import os.path
import toml
import click

pass_config = click.make_pass_decorator(dict)

DEFAULT_CONFIG = {
    'video_extension': 'avi',
    'converted_video_speed': 1,
    'calibration': {
        'animal_calibration': False,
        'calibration_init': None,
        'fisheye': False
    },
    'manual_verification': {
        'manually_verify': False
    },
    'triangulation': {
        'ransac': False,
        'optim': False,
        'scale_smooth': 2,
        'scale_length': 2,
        'scale_length_weak': 1,
        'reproj_error_threshold': 5,
        'score_threshold': 0.8,
        'n_deriv_smooth': 3,
        'constraints': [],
        'constraints_weak': []
    },
    'pipeline': {
        'videos_raw': 'videos-raw',
        'videos_raw_mp4': 'videos-raw-mp4',
        'pose_2d': 'pose-2d',
        'pose_2d_filter': 'pose-2d-filtered',
        'pose_2d_projected': 'pose-2d-proj',
        'pose_3d': 'pose-3d',
        'pose_3d_filter': 'pose-3d-filtered',
        'videos_labeled_2d': 'videos-labeled',
        'videos_labeled_2d_filter': 'videos-labeled-filtered',
        'calibration_videos': 'calibration',
        'calibration_results': 'calibration',
        'videos_labeled_3d': 'videos-3d',
        'videos_labeled_3d_filter': 'videos-3d-filtered',
        'angles': 'angles',
        'summaries': 'summaries',
        'videos_combined': 'videos-combined',
        'videos_compare': 'videos-compare',
        'videos_2d_projected': 'videos-2d-proj',
    },
    'filter': {
        'enabled': False,
        'type': 'medfilt',
        'medfilt': 13,
        'offset_threshold': 25,
        'score_threshold': 0.05,
        'spline': True,
        'n_back': 5,
        'multiprocessing': False
    },
    'filter3d': {
        'enabled': False,
        'medfilt': 17,
        'offset_threshold': 15
    }
}

def full_path(path):
    path_user = os.path.expanduser(path)
    path_full = os.path.abspath(path_user)
    path_norm = os.path.normpath(path_full)
    return path_norm

def load_config(fname):
    if fname is None:
        fname = 'config.toml'

    if os.path.exists(fname):
        config = toml.load(fname)
    else:
        config = dict()

    # put in the defaults
    if 'path' not in config:
        if os.path.exists(fname) and os.path.dirname(fname) != '':
            config['path'] = os.path.dirname(fname)
        else:
            config['path'] = os.getcwd()

    config['path'] = full_path(config['path'])

    if 'project' not in config:
        config['project'] = os.path.basename(config['path'])

    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
        elif isinstance(v, dict): # handle nested defaults
            for k2, v2 in v.items():
                if k2 not in config[k]:
                    config[k][k2] = v2

    return config

@click.group()
@click.version_option()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              help='The config file to use instead of the default "config.toml" .')
@click.pass_context
def cli(ctx, config):
    ctx.obj = load_config(config)

@cli.command()
@pass_config
def calibrate(config):
    from .calibrate import calibrate_all
    click.echo('Calibrating...')
    calibrate_all(config)

@cli.command()
@pass_config
def calibration_errors(config):
    from .calibration_errors import get_errors_all
    click.echo('Getting all the calibration errors...')
    get_errors_all(config)

@cli.command()
@click.option('--scorer', type=str, help='scorer to use for labeled data')
@pass_config
def tracking_errors(config, scorer=None):
    from .tracking_errors import get_tracking_errors
    click.echo('Comparing tracking to labeled data...')
    get_tracking_errors(config, scorer)
    
@cli.command()
@pass_config
def analyze(config):
    from .pose_videos import pose_videos_all
    click.echo('Analyzing videos...')
    pose_videos_all(config)

@cli.command()
@pass_config
def filter(config):
    from .filter_pose import filter_pose_all
    click.echo('Filtering tracked points...')
    filter_pose_all(config)

@cli.command()
@pass_config
def filter_3d(config):
    from .filter_3d import filter_pose_3d_all
    click.echo('Filtering tracked points...')
    filter_pose_3d_all(config)

@cli.command()
@pass_config
def triangulate(config):
    from .triangulate import triangulate_all
    click.echo('Triangulating points...')
    triangulate_all(config)


@cli.command()
@pass_config
def angles(config):
    from .compute_angles import compute_angles_all
    click.echo('Computing angles...')
    compute_angles_all(config)

@cli.command()
@pass_config
def summarize_3d(config):
    from .summarize import summarize_angles, summarize_pose3d, summarize_pose3d_filtered
    click.echo('Summarizing angles...')
    summarize_angles(config)

    click.echo('Summarizing 3D pose...')
    summarize_pose3d(config)

    if config['filter3d']['enabled']:
        click.echo('Summarizing 3D pose filtered...')
        summarize_pose3d_filtered(config)


@cli.command()
@pass_config
def summarize_2d(config):
    from .summarize import summarize_pose2d
    click.echo('Summarizing pose 2d...')
    summarize_pose2d(config)


@cli.command()
@pass_config
def summarize_2d_filter(config):
    from .summarize import summarize_pose2d_filtered
    click.echo('Summarizing pose 2d filtered...')
    summarize_pose2d_filtered(config)


@cli.command()
@pass_config
def summarize_errors(config):
    from .summarize import summarize_errors
    click.echo('Summarizing errors...')
    summarize_errors(config)


@cli.command()
@click.option('--nframes', default=200, type=int, show_default=True)
@click.option('--mode', default='bad', type=str, show_default=True)
@click.option('--no-pred', is_flag=True)
@click.option('--scorer', default=None, type=str)
@pass_config
def extract_frames(config, nframes=200, mode='bad', no_pred=False, scorer=None):
    from .extract_frames import extract_frames_picked, extract_frames_random
    click.echo('Extracting frames...')
    if no_pred:
        mode = 'random'
        extract_frames_random(config, nframes, scorer=scorer)
    else:
        extract_frames_picked(config, mode, nframes, scorer=scorer)


@cli.command()
@pass_config
def project_2d(config):
    from .project_2d import project_2d_all
    click.echo('Projecting 3D points back to 2D...')
    project_2d_all(config)


@cli.command()
@pass_config
def label_2d_proj(config):
    from .label_videos_proj import label_proj_all
    click.echo('Making 2D videos from 3D projections...')
    label_proj_all(config)
    
@cli.command()
@pass_config
def label_2d(config):
    from .label_videos import label_videos_all
    click.echo('Labeling videos in 2D...')
    label_videos_all(config)

@cli.command()
@pass_config
def label_2d_filter(config):
    from .label_videos import label_videos_filtered_all
    click.echo('Labeling videos in 2D...')
    label_videos_filtered_all(config)

@cli.command()
@pass_config
def label_3d(config):
    from .label_videos_3d import label_videos_3d_all
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)

@cli.command()
@pass_config
def label_3d_filter(config):
    from .label_videos_3d import label_videos_3d_filtered_all
    click.echo('Labeling videos in 3D...')
    label_videos_3d_filtered_all(config)

@cli.command()
@pass_config
def label_combined(config):
    from .label_combined import label_combined_all
    click.echo('Labeling combined videos...')
    label_combined_all(config)

@cli.command()
@pass_config
def label_filter_compare(config):
    from .label_filter_compare import label_filter_compare_all
    click.echo('Labeling videos to compare filtered vs raw tracking...')
    label_filter_compare_all(config)

@cli.command()
@pass_config
def draw_calibration(config):
    from .common import get_calibration_board_image
    import cv2
    click.echo('Drawing calibration board...')
    img = get_calibration_board_image(config)
    cv2.imwrite('calibration.png', img)

@cli.command()
@pass_config
def train_autoencoder(config):
    from .train_autoencoder import train_autoencoder
    click.echo('Training autoencoder...')
    train_autoencoder(config)


@cli.command()
@pass_config
def run_data(config):
    from .calibrate import calibrate_all
    from .pose_videos import pose_videos_all
    from .triangulate import triangulate_all
    from .compute_angles import compute_angles_all

    click.echo('Analyzing videos...')
    pose_videos_all(config)

    if config['filter']['enabled']:
        from .filter_pose import filter_pose_all
        click.echo('Filtering tracked points...')
        filter_pose_all(config)

    click.echo('Calibrating...')
    calibrate_all(config)

    click.echo('Triangulating points...')
    triangulate_all(config)

    if config['filter3d']['enabled']:
        from .filter_3d import filter_pose_3d_all
        click.echo('Filtering 3D points...')
        filter_pose_3d_all(config)
    
    click.echo('Computing angles...')
    compute_angles_all(config)


@cli.command()
@pass_config
def run_viz(config):
    from .label_videos import label_videos_filtered_all, label_videos_all
    from .label_videos_3d import label_videos_3d_all

    click.echo('Labeling videos in 2D...')
    if config['filter']['enabled']:
        label_videos_filtered_all(config)
    else:
        label_videos_all(config)
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)


@cli.command()
def visualizer():
    from .server import run_server
    run_server()

@cli.command()
@pass_config
def convert_videos(config):
    from .convert_videos import convert_all
    convert_all(config)

@cli.command()
@pass_config
def run_all(config):
    from .calibrate import calibrate_all
    from .pose_videos import pose_videos_all
    from .triangulate import triangulate_all
    from .compute_angles import compute_angles_all

    click.echo('Analyzing videos...')
    pose_videos_all(config)

    if config['filter']['enabled']:
        from .filter_pose import filter_pose_all
        click.echo('Filtering tracked points...')
        filter_pose_all(config)

    click.echo('Calibrating...')
    calibrate_all(config)

    click.echo('Triangulating points...')
    triangulate_all(config)

    if config['filter3d']['enabled']:
        from .filter_3d import filter_pose_3d_all
        click.echo('Filtering 3D points...')
        filter_pose_3d_all(config)

    click.echo('Computing angles...')
    compute_angles_all(config)

    from .label_videos import label_videos_filtered_all, label_videos_all
    from .label_videos_3d import label_videos_3d_all
    from .label_combined import label_combined_all

    click.echo('Labeling videos in 2D...')
    if config['filter']['enabled']:
        label_videos_filtered_all(config)
    else:
        label_videos_all(config)

    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)

    click.echo('Labeling combined videos...')
    label_combined_all(config)

if __name__ == '__main__':
    cli()
