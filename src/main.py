#!/usr/bin/env python3
# BombusCV: python OpenCV motion detection/recording tool developed for
# research on Bumblebees
# Copyright (C) 2022 Marco Radocchia
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see https://www.gnu.org/licenses/.

import cv2 as cv
import argparse
from argparse import ArgumentParser
from datetime import datetime as dt
from genericpath import isfile
from numpy import ndarray
from os import mkdir
from os.path import splitext, expanduser, isdir, join
from queue import Queue
from threading import Thread, Event
from typing import Tuple

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

# frames captured & queued, waiting to be processed
frames = Queue(10000)


def get_args() -> argparse.Namespace:
    argparser = ArgumentParser(allow_abbrev=False)
    argparser.add_argument(
        "-d",
        "--duration",
        type=int,
        metavar="<sec>",
        help=(
            "keep recording for <sec> seconds after motion detected"
            " (defaults to 3 seconds)"
        ),
        default=3,
    )
    argparser.add_argument(
        "-f",
        "--fps",
        type=int,
        metavar="<fps>",
        help="sets che fps for capture and video write",
        choices=[5, 10, 30, 60],
        default=10,
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
        default="720p",
    )
    argparser.add_argument(
        "-q", "--quiet", action="store_true", help="wheter to mute the output"
    )
    argparser.add_argument(
        "-v",
        "--video",
        type=str,
        metavar="<path_to_vid>",
        help="path of the video file",
    )
    return argparser.parse_args()


class FrameGrabber(Thread):
    """
    FrameGrabber thread class: frames are grabbed and stored in frames queue

    Parameters:
    -----------
    video: video file path, if None use camera input
    resolution: resolution of the video capture
    fps: framerate of the videocapture
    """

    def __init__(self, video: str, resolution: str, fps: int) -> None:
        Thread.__init__(self)
        self.terminate = Event()
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
            # define camera input
            self.cap = cv.VideoCapture(0)
            # get frame dimensions
            width, height = STD_DIMENSIONS[resolution]
            # set capture frame width & height
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            # set capture framerate
            self.cap.set(cv.CAP_PROP_FPS, fps)

    def run(self) -> None:
        """
        Run thread
        """
        global frames
        while self.cap.isOpened():
            _, frame = self.cap.read()
            frames.put(
                {
                    "date_time": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame": frame,
                }
            )

    def stop(self) -> None:
        """
        Stop thread safely releasing video capture
        """
        self.cap.release()
        self.terminate.set()


# main thread
class Main(Thread):
    """
    Main thread class

    Parameters:
    -----------
    cap: instance of cv.VideoCapture resource
    duration: integer number of seconds to keep recording after motion detected
    quiet: wheter to be quiet (no prints) or to be verbose (output prints)
    """

    def __init__(
        self, cap: cv.VideoCapture, duration: int, quiet: bool
    ) -> None:
        Thread.__init__(self)
        # keep recording for ``duration'' seconds after motion been detected
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
        video_format = self._get_video_format(filename)
        # set VideoWriter resolution & framerate based on video capture values
        dims = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self.writer = cv.VideoWriter(filename, video_format, self.fps, dims)
        # if not in quiet mode print resolution & framerate
        if not quiet:
            print(
                "Starting recording\n"
                "├─ Resolution: "
                f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}\n"
                f"└─ Frames per second: {int(self.fps)}"
            )
        self.prev_frame = None  # initialize previous frame to none

    def _get_video_format(self, filename: str) -> cv.VideoWriter_fourcc:
        _, ext = splitext(filename)
        # retrieve video based on file extension
        if ext in VIDEO_FORMAT:
            return VIDEO_FORMAT[ext]
        # default to mkv in case filename has no extension
        return VIDEO_FORMAT["mkv"]

    def _next_frame(self):
        # current frame becomes previous frame and next frame is pulled from
        # the ``frames'' queue
        self.prev_frame = self.frame
        self.frame = frames.get()

    def _motion_detected(self) -> Tuple[ndarray, ...]:
        proc_frame = cv.absdiff(self.prev_frame["frame"], self.frame["frame"])
        proc_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)
        proc_frame = cv.GaussianBlur(proc_frame, (21, 21), 0)
        proc_frame = cv.threshold(proc_frame, 30, 255, cv.THRESH_BINARY)[1]
        proc_frame = cv.dilate(proc_frame, None, iterations=3)
        contours, _ = cv.findContours(
            proc_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _write_frame(self) -> None:
        # frame it is copied to avoid writing date on the frame needed for
        # frame comparison in motion detection
        cloned_frame = self.frame["frame"].copy()
        # write date and time on frame before writing it to output file
        cloned_frame = cv.putText(
            cloned_frame,  # frame to write on
            self.frame["date_time"],  # displayed text
            (10, 40),  # position on frame
            cv.FONT_HERSHEY_DUPLEX,  # font
            1,  # font size
            (255, 255, 255),  # font color: white
            2,  # stroke
        )
        # write frame to output file
        self.writer.write(cloned_frame)

    def run(self) -> None:
        global frames
        # grab a couple of frames from the the que and enter main loop
        self.prev_frame = frames.get()
        self.frame = frames.get()
        # start writing loop which ends only when writer is released
        while self.writer.isOpened():
            if self._motion_detected():
                # if motion is detected record for <duration> seconds:
                # need to convert duration of recording in number of frames by
                # multiplying duration in seconds by frames per seconds value
                for _ in range(int(self.duration * self.fps)):
                    self._write_frame()
                    self._next_frame()
            else:
                # if motion is not detected keep pulling frames from queue
                self._next_frame()

    def stop(self) -> None:
        """
        Stop thread safely releasing video writer
        """
        self.writer.release()


def main() -> None:
    args = get_args()  # get command line arguments
    grabber = FrameGrabber(
        video=args.video, resolution=args.resolution, fps=args.fps
    )
    m = Main(cap=grabber.cap, quiet=args.quiet, duration=args.duration)
    try:
        grabber.start()
        m.start()
    except Exception:  # TODO: make grabber and main stop
        grabber.stop()
        m.stop()


if __name__ == "__main__":
    main()
