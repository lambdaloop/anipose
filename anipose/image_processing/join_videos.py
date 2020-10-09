import os
import os.path as op
from subprocess import call
import argparse


def join_videos(fnames, out_fname='tmp.mp4', copy=False, overwrite=False,
                verbose=True):
    """Join videos in time.

    Note: this does not check that they are adjacent in time.

    """
    if op.isfile(out_fname) and not overwrite:
        raise ValueError(f'File {out_fname} already exists')
    if verbose:
        print('Joining {}'.format(', '.join(fnames)))
    with open('tmp.txt', 'w') as fid:
        for fname in fnames:
            fid.write(f'file {fname}\n')
    call([f'ffmpeg -y -f concat -safe 0 -i tmp.txt '
          f'-c:v copy -c:a copy {out_fname}'],
         shell=True, env=os.environ)
    os.remove('tmp.txt')
    if not copy:
        for fname in fnames:
            os.remove(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', type=str, nargs='+',
                        help='The video filepaths')
    parser.add_argument('-of', '--out_fname', type=str,
                        help='The path to save the joined video to')
    parser.add_argument('-c', '--copy', action='store_true',
                        help='Whether copy the original file instead '
                        'of removing it')
    parser.add_argument('--overwrite', default=False, type=bool,
                        required=False, help='Whether to overwrite '
                        'existing file')
    parser.add_argument('--verbose', default=True, type=bool,
                        required=False, help='Whether to print '
                        'function updates')
    args = parser.parse_args()
    join_videos(fnames=args.fnames, out_fname=args.out_fname,
                copy=args.copy, overwrite=args.overwrite, verbose=args.verbose)
