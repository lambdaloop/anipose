import os
import os.path as op
from subprocess import call, run, PIPE, STDOUT
import json
import datetime
import argparse


# https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def get_length(fname):
    result = run(['ffprobe', '-v', 'error', '-show_entries',
                  'format=duration', '-of',
                  'default=noprint_wrappers=1:nokey=1', fname],
                 stdout=PIPE, stderr=STDOUT)
    return float(result.stdout)


def crop_video(fname, name=None, tmin=None, tmax=None, copy=False,
               overwrite=None, verbose=True):
    """Crop a video in the time domain."""
    if name == 'tmp':
        raise ValueError('Name `tmp` is reserved for internal use')
    ext = op.splitext(fname)[-1]
    out_fname = (fname if name is None or name == 'None'
                 else op.join(op.dirname(fname), name + ext))
    if op.isfile(out_fname) and not overwrite:
        raise ValueError(f'{out_fname} exists and overwrite is False')
    if out_fname == fname:  # move to tmp file if overwrite in place
        tmp_fname = out_fname
        out_fname = op.join(op.dirname(out_fname), 'tmp' + ext)
    else:
        tmp_fname = None
    if tmax is not None and tmax < 0:  # add time with negative
        tmax = get_length(fname) - tmax
    if verbose:
        tmin_str = 'beginning' if tmin is None else tmin
        tmax_str = 'end' if tmax is None else tmax
        print(f'Cropping {fname} from {tmin_str} to {tmax_str}')
    with open(fname.replace(ext, '.json'), 'r') as fid:
        json_data = json.load(fid)
    tmin_datetime = datetime.datetime.strptime(
        json_data['tmin'], '%H:%M:%S.%f')
    delta = datetime.timedelta(
        hours=tmin_datetime.hour, minutes=tmin_datetime.minute,
        seconds=tmin_datetime.second,
        microseconds=tmin_datetime.microsecond)
    if tmin is not None:
        tmin_delta = datetime.timedelta(seconds=tmin)
        json_data['tmin'] = str(delta + tmin_delta)
    if tmax is not None:
        tmax_delta = datetime.timedelta(seconds=tmax)
        if tmin is not None:
            tmax_delta -= tmin_delta
        json_data['tmax'] = str(delta + tmax_delta)
    if tmin < 0 or tmax is not None and tmax > get_length(fname):
        call(['ffmpeg -y -i {} -ss {} -to {} '
              '-c:v copy -c:a copy {}'.format(
                  json_data['video_file'], json_data['tmin'],
                  json_data['tmax'], out_fname)],
             shell=True, env=os.environ)
    else:
        tmax_str = '' if tmax is None else f'-to {tmax_delta}'
        call(['ffmpeg -y -i {} -ss {} {} '
              '-c:v copy -c:a copy {}'.format(
                  fname, tmin_delta, tmax_str, out_fname)],
             shell=True, env=os.environ)
    with open(out_fname.replace(ext, '.json'), 'w') as fid:
        fid.write(json.dumps(json_data, indent=4))
    if tmp_fname is not None:
        os.rename(out_fname, tmp_fname)
        os.rename(out_fname.replace(ext, '.json'),
                  tmp_fname.replace(ext, '.json'))
    elif not copy:
        os.remove(fname)
        os.remove(fname.replace(ext, '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str, help='The video filepath')
    parser.add_argument('name', type=str, help='The video name')
    parser.add_argument('tmin', type=float, default=None)
    parser.add_argument('tmax', type=float, nargs='?', default=None)
    parser.add_argument('-c', '--copy', action='store_true',
                        help='Whether copy the original file instead '
                        'of removing it')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Whether to overwrite an existing file')
    parser.add_argument('--verbose', default=True, type=bool,
                        required=False, help='Whether to print '
                        'function updates')
    args = parser.parse_args()
    crop_video(fname=args.fname, name=args.name, tmin=args.tmin,
               tmax=args.tmax, copy=args.copy, verbose=args.verbose,
               overwrite=args.overwrite)
