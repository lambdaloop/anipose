#!/usr/bin/env python3

import os, os.path
import toml
from calibrate_intrinsics import calibrate_intrinsics_all
from calibrate_extrinsics import calibrate_extrinsics_all

import click

## possible commands
# anipose calibrate # run calibration of intrinsics and extrinsics
# anipose analyze # analyze the poses for each video
# anipose label # create videos for each pose
# anipose run-data # run only the data portion (no viz)
# anipose run-viz # run only the visualization pipeline
# anipose run-all # run everything (run-data then run-viz)

pass_config = click.make_pass_decorator(dict)

DEFAULT_CONFIG = {
    'pipeline_videos_raw': 'videos-raw',
    'pipeline_pose_2d': 'pose-2d',
    'pipeline_pose_3d': 'pose-3d',
    'pipeline_videos_labeled_2d': 'videos-labeled',
    'pipeline_calibration': 'calibration',
    'pipeline_videos_labeled_3d': 'videos-3d'
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
        config['path'] = os.getcwd()

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
    click.echo('Calibrating...')
    calibrate_intrinsics_all(config)
    calibrate_extrinsics_all(config)

@cli.command()
@pass_config
def calibrate_intrinsics(config):
    click.echo('Calibrating intrinsics...')
    calibrate_intrinsics_all(config)

@cli.command()
@pass_config
def calibrate_extrinsics(config):
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
def triangulate(config):
    from triangulate import triangulate_all
    click.echo('Triangulating points...')
    triangulate_all(config)

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
    from pose_videos import pose_videos_all
    from triangulate import triangulate_all

    click.echo('Calibrating...')
    calibrate_intrinsics_all(config)
    calibrate_extrinsics_all(config)

    click.echo('Analyzing videos...')
    pose_videos_all(config)

    click.echo('Triangulating points...')
    triangulate_all(config)

    from label_videos import label_videos_all
    from label_videos_3d import label_videos_3d_all

    click.echo('Labeling videos in 2D...')
    label_videos_all(config)
    click.echo('Labeling videos in 3D...')
    label_videos_3d_all(config)

if __name__ == '__main__':
    cli()
