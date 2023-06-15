def crop_into_tiles(mosaic: str, out_file:str, n_tiles: int, idx_x:int, idx_y:int)->str:
    import os
    import math
    from osgeo import gdal

    # Functions
    def crop_pixels(input_file, output_file, window):
        # Window to crop by [left_x, top_y, width, height]
        translate_options = gdal.TranslateOptions(srcWin=window,
                                                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
                                                callback=gdal.TermProgress_nocb)
        gdal.Translate(output_file, input_file, options=translate_options)


    # idx_x number of the tile in the x dimension

    ds = gdal.Open(mosaic, 0)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    x_win_size = int(math.ceil(cols / n_tiles))
    y_win_size = int(math.ceil(rows / n_tiles))

    buffer = 10 # 10 pixels
    tile_count = 0
    
    idx_x = range(0, cols, x_win_size)[idx_x]
    idx_y = range(0, rows, y_win_size)[idx_y]

    if idx_y + y_win_size < rows:
        nrows = y_win_size
    else:
        nrows = rows - idx_y

    if idx_x + x_win_size < cols:
        ncols = x_win_size
    else:
        ncols = cols - idx_x

    win = [idx_x, idx_y, ncols, nrows]

    # Upper left corner
    win[0] = max(0, win[0] - buffer)
    win[1] = max(0, win[1] - buffer)

    w = win[2] + 2*buffer
    win[2] = w if win[0] + w < cols else cols - win[0]

    h = win[3] + 2*buffer
    win[3] = h if win[1] + h < cols else cols - win[1]

    crop_pixels(mosaic, out_file, win)
    tile_count += 1
    return out_file