#!/usr/bin/env python3
import cv2 as cv
import argparse
from argparse import ArgumentParser
from datetime import datetime as dt
from genericpath import isfile
from numpy import ndarray
from os import mkdir
from os.path import splitext, expanduser, isdir, join
from queue import Queue
from threading import Thread

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "360p": (480, 360),  # 16:9
    "480p": (640, 480),  # 4:3
    "720p": (1280, 720),  # 16:9
    "1080p": (1920, 1080),  # 16:9
    "4k": (3840, 2160),  # 16:9
}
# Video Encoding, might require additional installs,
# see http://www.fourcc.org/codecs.php
# NOTE: mkv reccommended formatd
VIDEO_FORMAT = {
    "avi": cv.VideoWriter_fourcc(*"XVID"),
    "mp4": cv.VideoWriter_fourcc(*"XVID"),
    "mkv": cv.VideoWriter_fourcc(*"XVID"),
}

# BUFFER: frame queued and in wait to be processed
frames = Queue(10000)


def get_video_format(filename: str) -> cv.VideoWriter_fourcc:
    _, ext = splitext(filename)
    # retrieve video based on file extension
    if ext in VIDEO_FORMAT:
        return VIDEO_FORMAT[ext]
    # default to mkv in case filename has no extension
    return VIDEO_FORMAT["mkv"]


def write_frame(out: cv.VideoWriter, frame: ndarray) -> None:
    # TODO: does the it need to be copied? it is copied to avoid writing date
    # on the frame needed for frame comparison in motion detection
    cloned_frame = frame["frame"].copy()
    # write date and time on frame before writing it to output file
    cloned_frame = cv.putText(
        cloned_frame,  # frame to write on
        frame["date_time"],  # displayed text
        (10, 40),  # position on frame
        cv.FONT_HERSHEY_DULPEX,  # font
        1,  # font size
        (255, 255, 255),  # font color: white
        2,  # stroke
    )
    # write frame to output file
    out.write(cloned_frame)


def set_cap_props(cap: cv.VideoCapture, res: str, fps: int) -> None:
    # get frame dimensions
    width, height = STD_DIMENSIONS[res]
    # set capture frame width & height
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    # set capture framerate
    cap.set(cv.CAP_PROP_FPS, fps)


def get_args() -> argparse.Namespace:
    argparser = ArgumentParser(allow_abbrev=False)
    argparser.add_argument(
        "-v",
        "--video",
        type=str,
        metavar="<path_to_vid>",
        help="path of the video file",
    )
    argparser.add_argument(
        "-r",
        "--resolution",
        type=str,
        metavar="<res>",
        help="resolution of the video capture",
        # valid resolution formats are the one listed in the global
        # STD_DIMENSIONS dictionary
        choices=list(STD_DIMENSIONS),
    )
    argparser.add_argument(
        "-q", "--quiet", action="store_true", help="wheter to mute the output"
    )
    argparser.add_argument(
        "-f",
        "--fps",
        type=int,
        metavar="<fps>",
        help="sets che fps for capture and video write",
        choices=[5, 10, 30, 60],
    )
    argparser.add_argument(
        "-d",
        "--duration",
        type=int,
        metavar="<sec>",
        help="keep recording for <sec> seconds after motion has been detected",
        default=3,
    )
    return argparser.parse_args()


def motion_detected(prev_frame, frame):
    proc_frame = cv.absdiff(prev_frame, frame)
    proc_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)
    proc_frame = cv.GaussianBlur(proc_frame, (21, 21), 0)
    proc_frame = cv.threshold(proc_frame, 30, 255, cv.THRESH_BINARY)[1]
    proc_frame = cv.dilate(proc_frame, None, iterations=3)
    contours, _ = cv.findContours(
        proc_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    return contours


# frame grabbing thread: frames are grabbed and stored in buffer
class ImageGrabber(Thread):
    def __init__(self, video: str, resolution: str, fps: int) -> None:
        Thread.__init__(self)
        # if video option is provided, then use video resource instead of
        # camera input
        if video:
            if isfile(video):
                # video input
                self.cap = cv.VideoCapture(expanduser(video))
            else:
                print("No video found")
                exit()
        else:
            # camera input
            self.cap = cv.VideoCapture(0)
        # default resolution
        resolution = resolution or "720p"
        # default fps
        fps = fps or 10
        set_cap_props(cap=self.cap, res=resolution, fps=fps)

    def run(self) -> None:
        global frames
        while self.cap.isOpened():
            _, frame = self.cap.read()
            frames.put(
                {
                    "date_time": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame": frame,
                }
            )


# main thread
class Main(Thread):
    def __init__(self, cap: cv.VideoCapture, quiet: bool, duration) -> None:
        Thread.__init__(self)
        # keep recording for ``duration'' seconds after motion has been
        # detected
        self.duration = duration
        # output video directory
        video_dir = join(expanduser("~"), "video")
        if not isdir(video_dir):  # if directory doesn't exits, create it
            mkdir(video_dir)
        # name file based on datetime for easier file organization & processing
        filename = join(
            video_dir, f"{dt.today().strftime('%Y-%m-%d_%H:%M:%S')}.mkv"
        )
        # retrieve video format chosen based on extension
        video_format = get_video_format(filename)
        # set VideoWriter resolution & framerate based on video capture values
        dims = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self.out = cv.VideoWriter(filename, video_format, self.fps, dims)
        # if not in quiet mode print resolution & framerate
        if not quiet:
            print(
                "Starting recording:\n"
                "Resolution: "
                f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}\n"
                f"Frames per second: {int(self.fps)}"
            )
        self.prev_frame = None  # initialize previous frame to none

    def _next_frame(self):
        # current frame becomes previous frame and next frame is pulled from
        # the ``frames'' queue
        self.prev_frame = self.frame
        self.frame = frames.get()

    def run(self) -> None:
        global frames  # TODO: is it needed?
        # grab a couple of frames from the the que and enter main loop
        self.prev_frame = frames.get()
        self.frame = frames.get()
        while True:
            if motion_detected(
                prev_frame=self.prev_frame["frame"], frame=self.frame["frame"]
            ):
                # if motion is detected record for <duration> seconds:
                # need to convert duration of recording in number of frames by
                # multiplying duration in seconds by frames per seconds value
                for _ in range(int(self.duration * self.fps)):
                    write_frame(out=self.out, frame=self.frame)
                    self._next_frame()
            else:
                # if motion is not detected keep pulling frames from queue
                self._next_frame()


def main() -> None:
    args = get_args()  # get command line arguments
    grabber = ImageGrabber(
        video=args.video, resolution=args.resolution, fps=args.fps
    )
    m = Main(cap=grabber.cap, quiet=args.quiet, duration=args.duration)
    grabber.start()
    m.start()
    grabber.join()
    m.join()


if __name__ == "__main__":
    main()
