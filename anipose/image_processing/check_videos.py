import os
import os.path as op
import cv2
import numpy as np
import json
import datetime
from subprocess import call
import argparse


def get_frames(frame_outputs, ext, width):
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


def plot_side_by_side(fnames, ext, title, width):
    caps = [cv2.VideoCapture(fname) for fname in fnames]
    wait_ms = np.round(100 / np.mean(
        [cap.get(cv2.CAP_PROP_FPS) for cap in caps])).astype(int)
    frame_outputs = [cap.read() for cap in caps]
    while all([fo[0] for fo in frame_outputs]):
        frames = get_frames(frame_outputs, ext, width)
        cv2.imshow(title, cv2.hconcat(frames))
        k = cv2.waitKey(wait_ms) & 0xff
        if k == ord('q'):
            break
        frame_outputs = [cap.read() for cap in caps]
    for cap in caps:
        cap.release()


def adjust_GUI(fnames, ext, title, width, verbose=True):
    # define GUI parameters for determining which stage
    n_videos = len(fnames)
    stages = ['align', 'start', 'stop', 'done']
    stage_idx = 0
    frame_indices = np.zeros((n_videos))
    video_idx = 0
    # display instructions
    print('First press left and right to move between videos and up and '
          'down to move between frames. Press `enter` when they are all '
          'aligned. Then use up and down to move to where the video '
          'should start and press `enter` again or press `left` to skip'
          'to the beginning. Finally move to where the video should end by '
          'pressing and press `enter` a third time or press `right` to skip '
          'to the end. Press `q` at any time to quit.')
    # get the input video
    caps = [cv2.VideoCapture(fname) for fname in fnames]
    frame_outputs = [cap.read() for cap in caps]
    # open files to write out to
    tmp_fnames = [fname.replace(ext, '_tmp.avi') for fname in fnames]
    outs = list()
    fpss = list()  # frames per seconds
    for fname, tmp_fname, cap in zip(fnames, tmp_fnames, caps):
        fps = cap.get(cv2.CAP_PROP_FPS)
        fpss.append(fps)
        if ext == '.mov':
            frame_width = int(cap.get(4))
            frame_height = int(cap.get(3))
        else:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
        outs.append(cv2.VideoWriter(
            tmp_fname, cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (frame_width, frame_height)))
    # initialize data for json sidecar
    tmins = None
    tmaxs = None
    # start the GUI
    while stages[stage_idx] != 'done':
        if verbose:
            print(f'Video times (on {video_idx}): ' + ', '.join(
                ['{:.4f}'.format(idx / fps) for idx, fps in
                 zip(frame_indices, fpss)]))
        # plot where we are
        frames = get_frames(frame_outputs, ext, width)
        cv2.imshow(title, cv2.hconcat(frames))
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
                    print('\nVideos starting at ' + ', '.join(
                        ['{:.4f}'.format(tmin) for tmin in tmins]) + '\n')
            elif stages[stage_idx] == 'stop':
                tmaxs = [idx / fps for idx, fps in zip(frame_indices, fpss)]
                if verbose:
                    print('\nVideos stopping at ' + ', '.join(
                        ['{:.4f}'.format(tmax) for tmax in tmaxs]) + '\n')
            stage_idx += 1
            if verbose:
                print(f'Changing to stage {stages[stage_idx]}')
        if key in (81, 83):
            if stages[stage_idx] == 'align':
                # move forward a video if right, back if left
                video_idx += key - 82
                video_idx = video_idx % n_videos
            if key == 81 and stages[stage_idx] == 'start':
                frame_indices -= frame_indices.min()
                tmins = [idx / fps for idx, fps in zip(frame_indices, fpss)]
                for cap, frame_i in zip(caps, frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                frame_outputs = [cap.read() for cap in caps]
                stage_idx += 1
                if verbose:
                    print('\nVideos starting at ' + ', '.join(
                        ['{:.4f}'.format(tmin) for tmin in tmins]) + '\n')
                    print(f'Changing to stage {stages[stage_idx]}')
            if key == 83 and stages[stage_idx] == 'stop':  # write until end
                while all([fo[0] and fo[1].size > 0 for fo in frame_outputs]):
                    frame_outputs = [cap.read() for cap in caps]
                    frame_indices += 1
                    for out, frame_output in zip(outs, frame_outputs):
                        out.write(frame_output[1])
                tmaxs = [idx / fps for idx, fps in zip(frame_indices, fpss)]
                stage_idx += 1
                if verbose:
                    print('\nVideos stopping at ' + ', '.join(
                        ['{:.4f}'.format(tmax) for tmax in tmaxs]) + '\n')
                    print(f'Changing to stage {stages[stage_idx]}')
        elif key in (82, 84):
            if stages[stage_idx] == 'align':  # move one video forward/back
                if key == 82:
                    frame_outputs[video_idx] = caps[video_idx].read()
                    if frame_outputs[video_idx][0]:
                        frame_indices[video_idx] += 1
                    else:  # went too far, go back to last frame
                        caps[video_idx].set(cv2.CAP_PROP_POS_FRAMES,
                                            frame_indices[video_idx])
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
                            for out, frame_output in zip(outs, frame_outputs):
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
    for out in outs:
        out.release()
    if tmins is None or tmaxs is None:  # not finished, delete and return
        for tmp_fname in tmp_fnames:
            os.remove(tmp_fname)
        return
    for fname, tmp_fname, tmin, tmax in zip(fnames, tmp_fnames, tmins, tmaxs):
        call([f'ffmpeg -y -i {tmp_fname} -c:v copy -c:a copy {fname}'],
             shell=True, env=os.environ)
        with open(fname.replace(ext, '.json'), 'r') as fid:
            json_data = json.load(fid)
        tmin_datetime = datetime.datetime.strptime(
            json_data['tmin'], '%H:%M:%S.%f')
        delta = datetime.timedelta(
            hours=tmin_datetime.hour, minutes=tmin_datetime.minute,
            seconds=tmin_datetime.second,
            microseconds=tmin_datetime.microsecond)
        tmin_delta = datetime.timedelta(seconds=tmin)
        json_data['tmin'] = str(delta + tmin_delta)
        tmax_delta = datetime.timedelta(seconds=tmax) - tmin_delta
        json_data['tmax'] = str(delta + tmax_delta)
        with open(fname.replace(ext, '.json'), 'w') as fid:
            fid.write(json.dumps(json_data, indent=4))
        os.remove(tmp_fname)


def check_videos(dirname, fbasename, width=1000, adjust=False,
                 verbose=True):
    """Display the videos from each camera simultaneously."""
    title, ext = op.splitext(fbasename)
    fnames = [op.join(dirname, f) for f in os.listdir(dirname)
              if f.endswith(fbasename)]
    if not fnames:
        raise ValueError('No files with `fbasename found in \n{}'.format(
                         ('\n'.join(os.listdir(dirname)))))
    if adjust:
        adjust_GUI(fnames, ext, title, width, verbose=verbose)
    else:
        plot_side_by_side(fnames, ext, title, width)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str,
                        help='The filepath to the directory with '
                        'the video files')
    parser.add_argument('fbasename', type=str,
                        help='The base name of the file within dirname')
    parser.add_argument('--width', type=float, default=1000,
                        help='The total width of the concatenated video '
                        ' in pixels')
    parser.add_argument('-a', '--adjust', action='store_true',
                        help='Whether to adjust the timing of file')
    parser.add_argument('--verbose', default=True, type=bool,
                        required=False, help='Whether to print '
                        'function updates')
    args = parser.parse_args()
    check_videos(dirname=args.dirname, fbasename=args.fbasename,
                 width=args.width, adjust=args.adjust,
                 verbose=args.verbose)
