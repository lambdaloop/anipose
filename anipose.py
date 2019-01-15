#!/usr/bin/env python3

import os, os.path
import toml
from common import full_path
import click

## possible commands
# anipose calibrate # run calibration of intrinsics and extrinsics
# anipose analyze # analyze the poses for each video
# anipose label # create videos for each pose
# anipose run_data # run only the data portion (no viz)
# anipose run_viz # run only the visualization pipeline
# anipose run_all # run everything (run_data then run_viz)

pass_config = click.make_pass_decorator(dict)

DEFAULT_CONFIG = {
    'pipeline_videos_raw': 'videos-raw',
    'pipeline_pose_2d': 'pose-2d',
    'pipeline_pose_2d_filter': 'pose-2d-filtered',
    'pipeline_pose_3d': 'pose-3d',
    'pipeline_videos_labeled_2d': 'videos-labeled',
    'pipeline_calibration_videos': 'calibration',
    'pipeline_calibration_results': 'calibration',
    'pipeline_videos_labeled_3d': 'videos-3d',
    'pipeline_angles': 'angles',
    'pipeline_summaries': 'summaries',

    'filter_enabled': True,
    'filter_medfilt': 13,
    'filter_offset_threshold': 25,
    'filter_score_threshold': 0.8,
    'filter_spline': True
}


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

    for k,v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v

    return config

@click.group()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              help='The config file to use instead of the default "config.toml".')
@click.pass_context
def cli(ctx, config):
    ctx.obj = load_config(config)

@cli.command()
@pass_config
def calibrate(config):
    from calibrate_intrinsics import calibrate_intrinsics_all
    from calibrate_extrinsics import calibrate_extrinsics_all
    click.echo('Calibrating...')
    calibrate_intrinsics_all(config)
    calibrate_extrinsics_all(config)

@cli.command()
@pass_config
def calibrate_intrinsics(config):
    from calibrate_intrinsics import calibrate_intrinsics_all
    click.echo('Calibrating intrinsics...')
    calibrate_intrinsics_all(config)

@cli.command()
@pass_config
def calibrate_extrinsics(config):
    from calibrate_extrinsics import calibrate_extrinsics_all
    click.echo('Calibrating extrinsics...')
    calibrate_extrinsics_all(config)

@cli.command()
@pass_config
def analyze(config):
    from pose_videos import pose_videos_all
    click.echo('Analyzing videos...')
    pose_videos_all(config)

@cli.command()
@pass_config
def filter(config):
    from filter_pose import filter_pose_all
    click.echo('Filtering tracked points...')
    filter_pose_all(config)
    
@cli.command()
@pass_config
def triangulate(config):
    from triangulate import triangulate_all
    click.echo('Triangulating points...')
    triangulate_all(config)


@cli.command()
@pass_config
def angles(config):
    from compute_angles import compute_angles_all
    click.echo('Computing angles...')
    compute_angles_all(config)

@cli.command()
@pass_config
def summarize(config):
    from summarize import summarize_angles, summarize_pose3d
    click.echo('Summarizing angles...')
    summarize_angles(config)

    click.echo('Summarizing 3D pose...')
    summarize_pose3d(config)


@cli.command()
@pass_config
def label_2d(config):
    from label_videos import label_videos_all
    click.echo('Labeling videos in 2D...')
    label_videos_all(config)

@cli.command()
@pass_config
def label_3d(config):
    from label_videos_3d import label_videos_3d_all
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)

@cli.command()
@pass_config
def run_data(config):
    from calibrate_intrinsics import calibrate_intrinsics_all
    from calibrate_extrinsics import calibrate_extrinsics_all
    from pose_videos import pose_videos_all
    from triangulate import triangulate_all

    click.echo('Calibrating...')
    calibrate_intrinsics_all(config)
    calibrate_extrinsics_all(config)

    click.echo('Analyzing videos...')
    pose_videos_all(config)

    click.echo('Triangulating points...')
    triangulate_all(config)

    click.echo('Running data processing')


@cli.command()
@pass_config
def run_viz(config):
    from label_videos import label_videos_all
    from label_videos_3d import label_videos_3d_all

    click.echo('Labeling videos in 2D...')
    label_videos_all(config)
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)


@cli.command()
@pass_config
def run_all(config):
    from calibrate_intrinsics import calibrate_intrinsics_all
    from calibrate_extrinsics import calibrate_extrinsics_all
    from pose_videos import pose_videos_all
    from triangulate import triangulate_all
    from compute_angles import compute_angles_all

    click.echo('Calibrating...')
    calibrate_intrinsics_all(config)
    calibrate_extrinsics_all(config)

    click.echo('Analyzing videos...')
    pose_videos_all(config)

    click.echo('Triangulating points...')
    triangulate_all(config)

    click.echo('Computing angles...')
    compute_angles_all(config)

    from label_videos import label_videos_all
    from label_videos_3d import label_videos_3d_all

    click.echo('Labeling videos in 2D...')
    label_videos_all(config)
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)

if __name__ == '__main__':
    cli()
