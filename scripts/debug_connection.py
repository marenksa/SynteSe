#!/usr/bin/env python3
"""Quick debug script to check Pupil Capture connection with timeouts."""
import sys

# Force unbuffered output
print("Script starting...", flush=True)

import zmq
import msgpack

print("Imports done.", flush=True)

TIMEOUT_MS = 5000  # 5 second timeout


def main() -> None:
    print("Connecting to Pupil Capture on port 50020...", flush=True)
    
    ctx = zmq.Context()

    # Connect to control port
    remote = ctx.socket(zmq.REQ)
    remote.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
    remote.setsockopt(zmq.SNDTIMEO, TIMEOUT_MS)
    remote.connect("tcp://127.0.0.1:50020")

    print("Socket connected, requesting SUB_PORT...", flush=True)

    try:
        # Get subscription port
        remote.send_string("SUB_PORT")
        sub_port = remote.recv_string()
        print(f"SUCCESS: Connected! SUB_PORT: {sub_port}", flush=True)
    except zmq.Again:
        print("ERROR: Timeout connecting to Pupil Capture.", flush=True)
        print("Make sure Pupil Capture is running!", flush=True)
        remote.close()
        ctx.term()
        return

    # Connect subscriber
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.subscribe(b"")  # Subscribe to everything

    print("\nListening for messages (5s timeout per message)...", flush=True)
    print("-" * 60, flush=True)

    received = 0
    gaze_count = 0
    frame_count = 0
    other_count = 0

    for i in range(20):
        try:
            parts = sub.recv_multipart()
            received += 1
            topic = parts[0].decode("utf-8")

            if topic.startswith("gaze"):
                gaze_count += 1
                if gaze_count <= 3:  # Only print first 3
                    try:
                        payload = msgpack.loads(parts[1], raw=False)
                        norm_pos = payload.get("norm_pos", (0, 0))
                        conf = payload.get("confidence", 0)
                        print(f"  GAZE: pos=({norm_pos[0]:.3f}, {norm_pos[1]:.3f}), confidence={conf:.2f}", flush=True)
                    except Exception as e:
                        print(f"  GAZE: (decode error: {e})", flush=True)
            elif topic.startswith("frame"):
                frame_count += 1
                if frame_count <= 2:  # Only print first 2
                    try:
                        payload = msgpack.loads(parts[1], raw=False)
                        w = payload.get("width", "?")
                        h = payload.get("height", "?")
                        fmt = payload.get("format", "?")
                        print(f"  FRAME: {w}x{h}, format={fmt}, parts={len(parts)}", flush=True)
                    except Exception as e:
                        print(f"  FRAME: (decode error: {e})", flush=True)
            else:
                other_count += 1
                if other_count <= 3:
                    print(f"  OTHER: topic={topic}", flush=True)

        except zmq.Again:
            print(f"\nTIMEOUT after {received} messages.", flush=True)
            break

    print("-" * 60, flush=True)
    print(f"Summary: {gaze_count} gaze, {frame_count} frame, {other_count} other", flush=True)

    if received == 0:
        print("\nNO DATA RECEIVED!", flush=True)
        print("Possible issues:", flush=True)
        print("  1. Pupil Capture is not streaming (check if eye cameras are working)", flush=True)
        print("  2. Frame Publisher plugin is not enabled (for video)", flush=True)
        print("  3. Gaze mapping is not running", flush=True)
    else:
        print("\nConnection is working!", flush=True)

    sub.close()
    remote.close()
    ctx.term()
    print("Done.", flush=True)



if __name__ == "__main__":
    main()
    sys.exit(0)
