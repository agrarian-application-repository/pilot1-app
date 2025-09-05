    "Read at native frame rate" or "Simulate Real-time": This crucial option makes FFmpeg read the input file at its native frame rate. Without -re, FFmpeg would read and process the file as fast as possible, which is usually too fast for live streaming protocols like RTMP, leading to buffer overflows on the server side. It simulates a live input.

-stream_loop -1

    "Loop Input Stream Indefinitely": This tells FFmpeg to loop the input file (rainbow_transition.mp4) forever. The -1 value means infinite looping. This is common for test streams or background content.

-i rainbow_transition.mp4

    "Input File": Specifies rainbow_transition.mp4 as the input video file.

-c:v libx264

    "Video Codec": Sets the video encoder to libx264, which is a highly efficient and widely supported H.264 video encoder. H.264 is the standard video codec for RTMP streaming.

-preset ultrafast

    "Encoding Speed Preset": libx264 has various presets that balance encoding speed against compression efficiency (file size/quality). ultrafast is the fastest preset, meaning it sacrifices some compression efficiency for very high encoding speed. This is usually desired for live streaming to minimize CPU usage and latency. Other presets include veryfast, fast, medium, slow, slower, veryslow.

-tune zerolatency

    "Tuning for Latency": This option tells libx264 to optimize for minimal latency. It disables features that might introduce delays (like certain frame reordering techniques) at the cost of slight compression efficiency. Crucial for real-time applications.

-s 1920x1080

    "Output Resolution (Size)": Sets the output video resolution to 1920 pixels wide by 1080 pixels high (Full HD). If your input video has a different resolution, FFmpeg will scale it.

-pix_fmt yuv420p

    "Pixel Format": Specifies the output pixel format as yuv420p. This is the most common and widely compatible pixel format for H.264 video, especially for web and streaming. It uses 4:2:0 chroma subsampling.

-r 30

    "Frame Rate": Sets the output video frame rate to 30 frames per second (fps). If your input has a different frame rate, FFmpeg will adjust it.

-g 60

    "GOP Size (Group of Pictures)": Defines the maximum interval between keyframes (IDR frames or I-frames). A keyframe is a complete image that doesn't rely on previous frames for decoding. A GOP of 60 means a keyframe will be inserted at least every 60 frames. For 30fps, this means a keyframe approximately every 2 seconds. This is important for stream seeking and playback stability, especially when clients join a stream in progress.

-keyint_min 60

    "Minimum Keyframe Interval": This sets the minimum interval between keyframes. It works in conjunction with -g. By setting it equal to -g, you are enforcing a fixed keyframe interval, meaning keyframes will occur exactly every 60 frames (if sc_threshold is 0).

-sc_threshold 0

    "Scene Change Threshold": This option controls whether FFmpeg (or libx264) should insert an extra keyframe when it detects a "scene change" (a sudden visual shift in the video). Setting it to 0 disables scene change detection. This means keyframes will only be inserted based on the -g and -keyint_min settings, providing a truly fixed GOP structure. This is often preferred for RTMP/HLS/DASH streaming for better compatibility and segmenting.

-b:v 2500k

    "Video Bitrate": Sets the target video bitrate to 2500 kilobits per second (kbps). This determines the quality and file size/bandwidth consumption of the video stream. 2500k is 2.5 Mbps, which is a common bitrate for 1080p video, balancing quality and bandwidth.

-maxrate 2500k

    "Maximum Video Bitrate": Specifies the maximum allowed video bitrate. It's often set equal to -b:v to prevent the bitrate from spiking too high during complex scenes, which can cause buffering issues for viewers.

-bufsize 5000k

    "Decoder Buffer Size": Sets the size of the decoder buffer. This buffer helps to smooth out bitrate fluctuations. A general rule of thumb is to set bufsize to at least 2 * maxrate. Here, 2 * 2500k = 5000k, so it's following that guideline.

-f flv

    "Output Format": Specifies the output container format as FLV (Flash Video). FLV is historically the most common container format for RTMP streaming.

"$rtmp_stream_url"

    "Output URL": This is the destination URL for the stream. It's an RTMP URL, like rtmp://10.91.222.62:1935/drone. FFmpeg will push the encoded video data to this address. The double quotes (") are good practice to ensure the variable content is treated as a single argument, even if it contained spaces (though RTMP URLs typically don't).