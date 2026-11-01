=====================================================
License Plate Recognition (LPR) Reference Application
=====================================================

Overview
========

``lpr_app`` reads license plates in a video stream on the Hailo-15. It runs the whole pipeline on the device:
the ISP processes the sensor frames, the DSP tiles the frame and crops the plates, the NN core runs a vehicle and
license-plate detector and an OCR model, and the encoder streams H.264 video. The plate texts are published as
metadata next to the video, and the analytic viewer on a host computer shows the video with the plates and their text.

The input is either a live camera (IMX678 sensor) or a raw NV12 video file, which makes it easy to show the
application on recorded footage.

Pipeline
--------

Each frame goes through these stages (the block that runs each stage is in brackets):

1. **ISP and multi-resize** [ISP, DSP]: the frame is resized to the encoded output streams and to a 1920×1080
   analytics stream.
2. **Tiling and detection** [DSP, NN core]: on every second frame, five tiles (the full frame and four overlapping
   quarters) are resized to 640×384, and YOLOv8n detects vehicles and license plates in them.
3. **Vehicle tracking and plate association** [CPU]: vehicles are tracked across frames. Plates are too small to
   track on their own; each plate takes the tracking ID of the vehicle that contains its center.
4. **Quality gate** [CPU]: only the best plate inside each vehicle, large enough and with a plate-like aspect ratio,
   is sent to the OCR. Vehicles whose plate text is already confirmed skip the OCR.
5. **Plate crop** [DSP]: the plate is cropped from the current detection and resized to 320×48, keeping its aspect
   ratio (left-aligned, gray padding).
6. **OCR** [NN core]: PP-OCRv6-medium reads the characters, in batches of up to 8 plates.
7. **Vote and publish** [CPU]: a plate text is published once it has been read 3 times (``--vote-count``) and holds
   the majority of the reads for that vehicle; the OCR then stops for that vehicle.
8. **Output** [encoder, CPU]: H.264 video over RTP/UDP, and the detections with their plate texts over ZeroMQ.

Requirements
============

* A Hailo-15H or Hailo-15L board with the Vision Processor Software 1.13.0 image.
* For live input, an IMX678 sensor module. File input needs no sensor.
* A Linux host connected to the board over Ethernet (by default the board is ``10.0.0.1`` and the host is
  ``10.0.0.2``), with Python 3 and GStreamer for the viewer, and ``ffmpeg`` to prepare video files.
* The Yocto SDK of the image, to build the application (see the cross-compilation guide in
  ``doc_sphinx/source/tools/cross_compile.rst``).

Build and Install
=================

The application changes the Hailo AI Analytics library and the OCR post-process, so all three components are built
and installed. Set ``-Dplatform=15l`` for a Hailo-15L board or ``-Dplatform=15h`` for a Hailo-15H board.

.. code-block:: sh

   source /path/to/yocto-sdk/environment-setup-aarch64-poky-linux

   # OCR post-process (libocr_post.so)
   cd hailo-postprocess
   meson setup build --buildtype=release --prefix=/usr -Dplatform=15l \
       -Dpost_processes_install_dir=/usr/lib/hailo-post-processes/
   ninja -C build
   DESTDIR=/tmp/lpr_install ninja -C build install
   cd ..

   # Hailo AI Analytics library and lpr_app
   cd hailo-analytics
   meson setup build --buildtype=release --prefix=/usr -Dplatform=15l -Dapps_install_dir=/home/root/apps
   ninja -C build -j 10 -l 10
   DESTDIR=/tmp/lpr_install ninja -C build install
   cd ..

Back up the files that the installation replaces, then copy the installation to the board:

.. code-block:: sh

   ssh root@10.0.0.1 'mkdir -p /home/root/lpr_backup && \
       cp -a /usr/lib/libhailo_analytics.so* /usr/lib/hailo-post-processes/libocr_post.so /home/root/lpr_backup/'
   scp -r /tmp/lpr_install/* root@10.0.0.1:/

Models
------

The application uses two models:

* **Detector**: ``hailo_yolov8n_384_640.hef``, included in the board image under
  ``/home/root/apps/shared/resources/``.
* **OCR**: ``paddle_ocr_v6_medium_recognition.hef``, PP-OCRv6-medium quantized for license plates. It is not part
  of the 1.13.0 release and is provided separately. Copy it to the board:

  .. code-block:: sh

     scp paddle_ocr_v6_medium_recognition.hef root@10.0.0.1:/home/root/apps/license_plate_recognition/resources/

Run the Viewer on the Host
==========================

Set up the analytic viewer as described in ``doc_sphinx/source/hailo_analytics/apps/analytic_viewer.rst``, then start
the LPR viewer before the application:

.. code-block:: sh

   cd tools/analytic_viewer
   python3 lpr_app_viewer.py

The viewer receives the video on UDP port 5000 and the metadata from ``tcp://10.0.0.1:7000``, and shows the video
with the vehicle and plate boxes and a side panel with the recognized plates.

The application sends two video streams: the full-resolution stream to UDP port 5000 and a 1280×720 stream to
port 5002. To view the smaller stream, add ``-u 5002``.

To record the annotated video, for example for a demonstration, add ``--save-mkv``:

.. code-block:: sh

   python3 lpr_app_viewer.py -u 5002 --save-mkv lpr_demo.mkv

Run with a Live Camera
======================

On a newly flashed board, select the configuration for the platform and the sensor once. The script detects the
Hailo-15H or Hailo-15L and the connected sensor, and points ``/etc/imaging/cfg/medialib_configs`` to the matching
configuration. Without this step, ``lpr_app`` stops with ``Configuration architecture ... does not match current
architecture``.

.. code-block:: sh

   ssh root@10.0.0.1 setup_hailo_sensor.sh

Then run the application:

.. code-block:: sh

   ssh root@10.0.0.1
   cd /home/root/apps/license_plate_recognition
   ./lpr_app -t 3600 -p

``-t`` is the run time in seconds; stop earlier with Ctrl+C. ``-p`` prints the frame rate of the encoded stream
every few seconds (about 30 FPS, drop rate 0).

Run from a Video File
=====================

The application can read a raw NV12 file instead of the camera, and plays it in a loop.

1. On the host, convert the video to raw NV12 at 1920×1080 and 30 FPS (the video is scaled to fit and padded, so its
   aspect ratio is kept):

   .. code-block:: sh

      ffmpeg -i input.mp4 -t 20 \
          -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" \
          -pix_fmt nv12 -f rawvideo clip.nv12

   A raw 1080p frame takes about 3 MB, so 20 seconds of video take about 1.9 GB. Check the free space on the board
   with ``df -h /home/root`` and shorten the clip with ``-t`` if needed.

2. Copy the file to the board:

   .. code-block:: sh

      scp clip.nv12 root@10.0.0.1:/home/root/

3. Start the viewer on the host (see above), then run the application:

   .. code-block:: sh

      cd /home/root/apps/license_plate_recognition
      ./lpr_app -f /home/root/clip.nv12 -W 1920 -H 1080 --fps 30 -t 300 -p

Command-Line Options
====================

=================================  =====================================================================================
Option                             Meaning
=================================  =====================================================================================
``-t``, ``--timeout``              Run time in seconds (default 60)
``-p``, ``--print-fps``            Print the frame rate of the encoded stream
``-o``, ``--host-ip``              Destination of the video streams (default: the computer connected over SSH,
                                   otherwise ``10.0.0.2``)
``-u``, ``--udp-port``             UDP port of the full-resolution stream (default 5000); the 1280×720 stream uses
                                   this port + 2
``-z``, ``--zmq-port``             Port of the metadata publisher (default 7000)
``-f``, ``--file``                 Raw NV12 video file to play instead of the camera
``-W``, ``-H``, ``--fps``          Width, height and frame rate of the video file (required with ``--file``)
``-m``, ``--tracking-mode``        Vehicle tracker: ``slow``, ``balanced`` (default) or ``fast``
``--vote-count``                   Matching OCR reads needed before a plate text is published (default 3)
``--ocr-hef``                      OCR HEF to use instead of the bundled PP-OCRv6-medium HEF
``-c``, ``--config-file-path``     Media Library configuration file
``-a``, ``--profile``              Media Library profile
``--use-medialib-service``         Connect to ``media_library_service`` (camera input only)
=================================  =====================================================================================

Output Metadata
===============

For every frame, the application publishes a ``hailo_analytics.Frame`` protobuf message over ZeroMQ
(``tcp://<board-ip>:7000``). It contains the vehicle and license-plate detections with their boxes, confidence and
tracking ID. A plate whose text is confirmed carries a classification of type ``ocr`` with the text and its
confidence. The message format is defined in
``hailo-analytics/hailo_analytics_api/src/pipeline/codecs/protos/analytics_metadata.proto``, and
``tools/analytic_viewer/lpr_app_viewer.py`` shows how to read it.

Customization
=============

* ``--vote-count``: fewer reads publish more plates, sooner, but also more wrong texts; more reads do the opposite.
* ``--tracking-mode``: ``fast`` drops lost vehicles sooner; ``slow`` keeps them longer through occlusions.
* The OCR reads any text and does not know plate formats. For a known regional format, accuracy can be raised by
  correcting look-alike characters by position (for example forcing digits where the format has digits).
* The detection, quality-gate, crop and OCR settings are set in ``lpr_pipeline_builder.cpp``.

Limitations
===========

* Two-line plates and motorcycle plates are not supported.
* A plate is read only when it lies inside a detected vehicle.
* Small plates (less than about 24 pixels high in the 1080p analytics stream) are read much less reliably.
* Plates that are visible only briefly may not collect enough matching reads to be published.
* The OCR can swap look-alike characters, such as 0 and O, 1 and I, 5 and S, or 8 and B.
