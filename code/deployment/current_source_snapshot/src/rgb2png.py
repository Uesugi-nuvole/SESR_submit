#!/usr/bin/env python3
import os
import argparse
from PIL import Image


def restore_image_from_raw_rgb(input_file, output_file, width=1280, height=720):
    if not os.path.exists(input_file):
        print(f"Error: File not found - {input_file}")
        return -1

    print(f"Reading raw RGB file: {input_file}")

    with open(input_file, "rb") as f:
        data = f.read()

    expected_bytes = width * height * 3
    actual_bytes = len(data)

    print(f"Read {actual_bytes} bytes.")
    print(f"A {width}x{height} RGB888 image requires {expected_bytes} bytes.")

    if actual_bytes == 0:
        print("Error: input file is empty.")
        return -1

    if actual_bytes < expected_bytes:
        print(f"Warning: insufficient data.")
        print(f"Only {actual_bytes} bytes found, need {expected_bytes} bytes.")
        print(f"Remaining pixels will be filled with black.")
        data = data + bytes(expected_bytes - actual_bytes)

    elif actual_bytes > expected_bytes:
        print(f"Warning: input data is larger than expected.")
        print(f"Extra {actual_bytes - expected_bytes} bytes will be ignored.")
        data = data[:expected_bytes]

    rgb_data = []

    for i in range(0, expected_bytes, 3):
        r = data[i]
        g = data[i + 1]
        b = data[i + 2]
        rgb_data.append((r, g, b))

    img = Image.new("RGB", (width, height))
    img.putdata(rgb_data)
    img.save(output_file)

    print(f"Success! Image saved as: {output_file}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore RGB888 raw data to PNG image.")
    parser.add_argument("input", help="input raw RGB file")
    parser.add_argument("-o", "--output", default="sr_img.png", help="output image file")
    parser.add_argument("--width", type=int, default=1280, help="image width")
    parser.add_argument("--height", type=int, default=720, help="image height")

    args = parser.parse_args()

    restore_image_from_raw_rgb(
        args.input,
        args.output,
        width=args.width,
        height=args.height
    )
