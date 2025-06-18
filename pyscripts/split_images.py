def main():
    W = 1920
    H = 1080

    tile_sizes = [i for i in range(120, 640, 4)]
    overlaps = [i for i in range(20, 320, 4)]

    for tile_size in tile_sizes:
        for overlap in overlaps:

            try:
                num_tiles_x = (W - overlap) / (tile_size - overlap)
                num_tiles_y = (H - overlap) / (tile_size - overlap)

                # print(f"testing {tile_size}, {overlap}")
                if num_tiles_x.is_integer() and num_tiles_y.is_integer() and overlap <= tile_size / 2:
                    print("FOUND")
                    print(f"testing {tile_size}, {overlap}")
                    print(num_tiles_x, num_tiles_y)
                    print(tile_size, overlap)
                    print(f"memory incr = {(num_tiles_x * num_tiles_y * tile_size * tile_size) / (H * W)}")

            except ZeroDivisionError:
                pass


if __name__ == "__main__":
    main()

"""
FOUND
testing 120, 60
31.0 17.0
120 60
memory incr = 3.6597222222222223
FOUND
testing 156, 72
22.0 12.0
156 72
memory incr = 3.098333333333333
FOUND
testing 240, 72
11.0 6.0
240 72
memory incr = 1.8333333333333333
FOUND
testing 240, 100
13.0 7.0
240 100
memory incr = 2.5277777777777777
FOUND
testing 240, 120
15.0 8.0
240 120
memory incr = 3.3333333333333335
FOUND
testing 520, 240
6.0 3.0
520 240
memory incr = 2.3472222222222223
"""





