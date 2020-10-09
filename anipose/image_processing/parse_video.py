import os
import os.path as op
import cv2
import numpy as np
import re
import json
import datetime
from subprocess import call
import argparse


def _get_av_offset(fname):
    """Get the offset between the audio and the video for the clip."""
    from subprocess import run, PIPE, STDOUT  # run used in another context
    result = run(['ffprobe', '-show_entries',
                  'stream=codec_type,start_time', '-v', '0', '-of',
                  'compact=p=1:nk=0', fname],
                 stdout=PIPE, stderr=STDOUT)
    output = result.stdout.decode('utf-8').split('\n')
    assert output[0].startswith('stream|codec_type=video|start_time')
    assert output[1].startswith('stream|codec_type=audio|start_time')
    return float(output[0].strip('stream|codec_type=video|start_time')) - \
        float(output[1].strip('stream|codec_type=audio|start_time'))


def _get_frames(frame_outputs, ext, width):
    # make the whole thing 1000 pixels wide
    scale = width / (max(frame_outputs[0][1].shape) * len(frame_outputs))
    frames = list()
    for ret, frame in frame_outputs:
        if ext == '.mov':
            frame = frame.swapaxes(0, 1)
            frame = frame[:, ::-1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frames.append(frame)
    return frames


def _init_out_files(fnames, ext, caps, fpss, tmp_fnames):
    out_writers = list()
    for fname, cap, fps, tmp_fname in zip(fnames, caps, fpss, tmp_fnames):
        if ext == '.mov':
            frame_width = int(cap.get(4))
            frame_height = int(cap.get(3))
        else:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
        out_writers.append(cv2.VideoWriter(
            tmp_fname, cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (frame_width, frame_height)))
    return out_writers


def parse_video(fnames, cameras, sub, ses=None, run=None, acq=None,
                out_dir=None, width=1000, starting_time=0, verbose=True):
    """Split a video up into component parts synchronized across cameras."""
    if len(cameras) != len(fnames):
        raise ValueError('There must be the same number of camera names and '
                         f'fnames, got {len(cameras)} cameras and '
                         f'{len(fnames)} fnames')
    if isinstance(starting_time, list):
        if len(starting_time) == 1:
            starting_time = starting_time[0]
        elif len(starting_time) != len(cameras):
            raise ValueError(f'{len(fnames)} starting times may be provided '
                             f'for each video but got {len(starting_time)}')
    _, ext = op.splitext(fnames[0])
    for fname in fnames:
        if not op.isfile(fname):
            raise ValueError(f'{fname} file not found')
        if op.splitext(fname)[-1] != ext:
            raise ValueError(f'Video types do not match, got {ext} and '
                             f'{op.splitext(fname)[-1]}')
    ext = ext.lower()
    # define the out name template
    bids_name = f'sub-{sub}'
    for key, value in {'ses': ses, 'run': run, 'acq': acq}.items():
        if value is not None:
            bids_name += f'_{key}-{value}'
    if out_dir is None:
        out_dir = './'
    if ses is not None:
        out_dir = op.join(out_dir, f'ses-{ses}')
    # make the output directory
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
    if not op.isdir(op.join(out_dir, 'videos-raw')):
        os.makedirs(op.join(out_dir, 'videos-raw'))
    if not op.isdir(op.join(out_dir, 'calibration')):
        os.makedirs(op.join(out_dir, 'calibration'))
    # define GUI parameters for determining which stage
    n_videos = len(fnames)
    stages = ['align', 'start', 'stop']
    # display instructions
    print('First press left and right to move between videos and up and '
          'down to move between frames. Press `enter` when they are all '
          'aligned. Then use up and down to move to where the video '
          'should start and press `enter` again. Then move to where the '
          'video should end by pressing up and press `enter` a third time. '
          'You can press right during the stop stage to go forward a '
          'minute (you can crop after). Follow the prompts to name and tag '
          'the video and repeat. Press `q` at any time to quit.')
    # get the input video
    video_idx = 0
    stage_idx = 0
    caps = [cv2.VideoCapture(fname) for fname in fnames]
    fpss = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    if isinstance(starting_time, list):
        frame_indices = (np.array(starting_time) * min(fpss)).astype(int)
    else:
        frame_indices = np.ones((n_videos), dtype=int) \
            * int(min(fpss)) * starting_time
    for cap, frame_index in zip(caps, frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_outputs = [cap.read() for cap in caps]
    tmp_fnames = [op.join(out_dir, op.basename(fname.lower()).replace(
                  ext, '_tmp.avi')) for fname in fnames]
    out_writers = _init_out_files(fnames, ext, caps, fpss, tmp_fnames)

    # initialize data for json sidecar
    tmins = tmaxs = None
    # start the GUI
    while True:
        if verbose:
            print(f'Video times (on {video_idx}): ' + ' '.join(
                ['{:.4f}'.format(idx / fps) for idx, fps in
                 zip(frame_indices, fpss)]))
        # plot where we are
        frames = _get_frames(frame_outputs, ext, width)
        cv2.imshow('Video Alignment Editor', cv2.hconcat(frames))
        # pull for key
        key = cv2.waitKey(0)
        while key not in (81, 82, 83, 84, 13, ord('q')):
            key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == 13:
            if stages[stage_idx] == 'start':
                tmins = [idx / fps for idx, fps in zip(frame_indices, fpss)]
                if verbose:
                    print('\nVideos starting at ' + ' '.join(
                        ['{:.4f}'.format(tmin) for tmin in tmins]) + '\n')
            elif stages[stage_idx] == 'stop':
                tmaxs = [idx / fps for idx, fps in zip(frame_indices, fpss)]
                if all([tmax == tmin for tmax, tmin in zip(tmaxs, tmins)]):
                    break  # if nothing was written skip
                if verbose:
                    print('\nVideos stopping at ' + ' '.join(
                        ['{:.4f}'.format(tmax) for tmax in tmaxs]) + '\n')
                # open new files
                for out in out_writers:
                    out.release()
                # get user input
                ok = False
                while not ok:
                    name = None
                    while not name:  # no blank names
                        name = input('Segment name? ')
                    if name == 'delete':  # don't write if delete is given
                        break
                    out_fnames = [op.join(out_dir, ('calibration' if 'calib' in
                                                    name else 'videos-raw'),
                                          f'camera-{camera}_{bids_name}_'
                                          f'seg-{name}{ext}')
                                  for camera, fname in zip(cameras, fnames)]
                    if any([op.isfile(fname) for fname in out_fnames]):
                        if input('Overwrite (y/N)? ').lower() == 'y':
                            ok = True
                    else:
                        ok = True
                if ok:
                    # commas, spaces or tabs are okay
                    tags = [tag.strip() for tag in
                            re.split(' |,|\t', input('Tags? ')) if tag.strip()]
                    # save
                    for fname, out_fname, tmp_fname, tmin, tmax in zip(
                            fnames, out_fnames, tmp_fnames, tmins, tmaxs):
                        call([f'ffmpeg -y -i {tmp_fname} -c:v copy -c:a copy '
                              f'{out_fname}'], shell=True, env=os.environ)
                        json_data = dict(
                            video_file=op.abspath(fname), tags=tags, ses=ses,
                            tmin=str(datetime.timedelta(seconds=tmin)),
                            tmax=str(datetime.timedelta(seconds=tmax)))
                        with open(out_fname.replace(ext, '.json'), 'w') as fid:
                            fid.write(json.dumps(json_data, indent=4))
                        os.remove(tmp_fname)
                # reset for next video
                tmins = tmaxs = None
                out_writers = _init_out_files(fnames, ext, caps, fpss,
                                              tmp_fnames)
            stage_idx += 1
            stage_idx = stage_idx % len(stages)
            if verbose:
                print(f'Changing to stage {stages[stage_idx]}')
        if key in (81, 83):
            if stages[stage_idx] == 'align':
                # move forward a video if right, back if left
                video_idx += key - 82
                video_idx = video_idx % n_videos
            elif stages[stage_idx] == 'stop' and key == 83:
                n_frames = np.round(min(fpss) * 60).astype(int)
                for _ in range(n_frames):
                    frame_outputs = [cap.read() for cap in caps]
                    if all([fo[0] and fo[1].size > 0 for fo in
                            frame_outputs]):  # none too far
                        frame_indices += 1
                        for out, frame_output in zip(out_writers,
                                                     frame_outputs):
                            out.write(frame_output[1])
            elif stages[stage_idx] == 'start':
                frame_indices_saved = frame_indices.copy()
                frame_indices += (key - 82) * np.round(min(fpss) * 60
                                                       ).astype(int)
                for cap, frame_i in zip(caps, frame_indices):
                    ret = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                    if not ret:  # put back
                        frame_indices = frame_indices_saved
                        for cap, frame_i in zip(caps, frame_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                        break
                frame_outputs = [cap.read() for cap in caps]
        elif key in (82, 84):
            if stages[stage_idx] == 'align':  # move one video forward/back
                if key == 82:
                    frame_outputs[video_idx] = caps[video_idx].read()
                    if frame_outputs[video_idx][0]:
                        frame_indices[video_idx] += 1
                    else:  # went too far, go back to last frame
                        caps[video_idx].set(cv2.CAP_PROP_POS_FRAMES,
                                            frame_indices[video_idx])
                        frame_outputs[video_idx] = caps[video_idx].read()
                else:  # key == 84 (down -> go back)
                    if frame_indices[video_idx] - 1 >= 0:
                        frame_indices[video_idx] -= 1
                        caps[video_idx].set(cv2.CAP_PROP_POS_FRAMES,
                                            frame_indices[video_idx])
                        frame_outputs[video_idx] = caps[video_idx].read()
            else:
                if key == 82:  # start and stop can both go forward
                    frame_outputs = [cap.read() for cap in caps]
                    if all([fo[0] and fo[1].size > 0 for fo in
                            frame_outputs]):  # none too far
                        frame_indices += 1
                        if stages[stage_idx] == 'stop':
                            # write as we go for stop
                            for out, frame_output in zip(out_writers,
                                                         frame_outputs):
                                out.write(frame_output[1])
                    else:
                        for cap, frame_i in zip(caps, frame_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                elif key == 84 and stages[stage_idx] == 'start':
                    if (frame_indices - 1).min() >= 0:
                        frame_indices -= 1  # only start can go back
                        for cap, frame_i in zip(caps, frame_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                        frame_outputs = [cap.read() for cap in caps]
    for out in out_writers:
        out.release()
    for tmp_fname in tmp_fnames:
        if op.isfile(tmp_fname):
            os.remove(tmp_fname)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', type=str, nargs='+',
                        help='The filepath of the video')
    parser.add_argument('-c', '--cameras', type=str, nargs='+',
                        help='The names of the cameras')
    parser.add_argument('-s', '--sub', type=str, help='The subject id',
                        required=True)
    parser.add_argument('-n', '--ses', type=str, help='The session id',
                        default=None, required=False)
    parser.add_argument('-r', '--run', type=str, help='The run id',
                        default=None, required=False)
    parser.add_argument('-a', '--acq', type=str, help='The acquisition id',
                        default=None, required=False)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='The path to save the joined video to')
    parser.add_argument('--width', type=float, default=1000,
                        help='The total width of the concatenated video '
                        ' in pixels')
    parser.add_argument('--starting_time', type=float, nargs='+', default=0,
                        help='Which time at which to start')
    parser.add_argument('--verbose', default=True, type=bool,
                        required=False, help='Whether to print '
                        'function updates')
    args = parser.parse_args()
    parse_video(fnames=args.fnames, cameras=args.cameras, sub=args.sub,
                ses=args.ses, run=args.run, acq=args.acq, out_dir=args.out_dir,
                width=args.width, starting_time=args.starting_time,
                verbose=args.verbose)
