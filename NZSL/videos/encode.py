# -*- coding: utf-8 -*-
"""
Converts video into spikes using retina-inspired encoding.
Subsequent frames are compared with each other pixel by pixel.
A frame is split into blocks, imitating peripheral vision.
These blocks are converted into greyscale to compare brightness.
The focus of the image is defined as the block with most activity.
This focus block is assessed in colour.

How spikes are created in peripheral blocks:
When a pixel's brightness changes more than a defined threshold between frames,
a spike is created for this pixel location. If there are more than a defined
number of spikes in a block, the block creates a spike.

How spikes are created in the focus block:
For each pixel, the BGR values are compared. If their sum differs by more than
a defined threshold, this pixel creates a spike.

@author: Anne Wendt
"""

import os
import sys
import time

import colour.difference as cd
import cv2  # OpenCV library
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
###                            GLOBAL PARAMETERS                            ###
###############################################################################

# DIRECTORY = os.path.join('H:', os.sep, 'Data', 'Anne', 'projects', 'NZSL', 'videos')
# DIRECTORY = os.path.join('C:', os.sep, 'Users', 'em165153', 'Documents', 'JESTER')
DIRECTORY = 'samples'
DIRECTORY2 = 'encoded_samples'
FILETYPE = '.mp4'  # only these files in the directory will be encoded

TEST_RUN = False  # play videos more slowly and do not close windows at the end

# this threshold is used when comparing subsequent frames on a pixel base
# if they differ enough, a spike is created
PIXEL_THRESHOLD = 3  # applies to the outer blocks in greyscale = intensity difference
FOVEA_THRESHOLD = 5  # applies to the central block in colour = DELTA E JND

# how many blocks do we want in each column and row for each block level
#            c  r
# BLOCKS = [[3, 3],  # block 0 (outermost periphery)
#           [6, 5],  # block 1
#           [5, 4],  # block 2
#           [7, 5],  # block 3
#           [8, 8]]  # block 4 (fovea)

# BLOCKS = [[10, 6], [15, 10], [23, 15], [35, 23], [55, 36], [84, 56], [94, 85]]  # tal orig
# BLOCKS = [[6, 4], [8, 6], [12, 9], [17, 13], [30, 20], [34, 30]]  # tal by 2
# BLOCKS = [[4, 4], [6, 5], [7, 10], [16, 12], [18, 16]]  # tal by 3
# BLOCKS = [[3, 2], [4, 4], [6, 5], [9, 7], [12, 11]]  # tal by 4
# BLOCKS = [[3, 3], [5, 4], [7, 5], [8, 8]]  # tal by 5
# BLOCKS = [[2, 2], [4, 3], [6, 4], [6, 6]]  # tal by 6
# BLOCKS = [[2, 2], [3, 2], [4, 3], [5, 5]]  # tal by 7
# BLOCKS = [[3, 2], [3, 3], [4, 4]]  # tal by 8
# BLOCKS = [[2, 2], [3, 2], [4, 3]]  # tal by 9
# BLOCKS = [[3, 1], [2, 2], [3, 3]]  # tal by 10
# BLOCKS = [[9, 6], [15, 10], [23, 16], [36, 25], [64, 37], [95, 66], [106, 96]]  # mni times 2
# BLOCKS = [[5, 4], [8, 6], [15, 9], [21, 15], [33, 22], [38, 34]]  # mni orig
# BLOCKS = [[3, 2], [6, 4], [7, 5], [12, 8], [13, 12]]  # mni by 2
# BLOCKS = [[3, 2], [5, 3], [6, 5], [7, 6]]  # mni by 3
BLOCKS = [[2, 2], [3, 2], [3, 3], [5, 4]]  # mni by 4
# BLOCKS = [[2, 2], [2, 2], [4, 3]]  # mni by 5

# threshold for a periphery block to be counted as spike
#BLOCK_THRESHOLD = 0.0

# threshold for a fovea block to be counted as spike
#BLOCK_FOVEA_THRESHOLD = 0.0

# how many times smaller or larger than its direct neighbours is each block
BLOCK_SCALING_FACTOR = 4 ** (1 / (len(BLOCKS) - 1))
if TEST_RUN:
    print("Block scaling factor:", BLOCK_SCALING_FACTOR)

# parameters that will be used to draw the blocks
# OpenCV uses BGR colour format!
COLOURS = [#(128, 0, 0),  # level -1 - dark blue
           (255, 0, 0),  # level 0 - blue
#           (255, 255, 0),  # level 1 - cyan
           (0, 255, 0),  # level 2 - green
           (0, 255, 255),  # level 3 - yellow
           (0, 128, 255),  # level 4 - orange
           (0, 0, 255)]  # level 5 - red

LINE_WIDTH = 1  # in pixels


###############################################################################
###                                FUNCTIONS                                ###
###############################################################################

def clean_up_and_exit(capture, exit_message):
    """
    Releases the video capture, closes all open windows
    and exits the program with a message.
    """

    capture.release()
    if not TEST_RUN:
        cv2.destroyAllWindows()
    sys.exit(exit_message)


def draw_boundaries(frame, block_info, boundaries):
    """
    Draws the block boundaries onto the frame.
    """

    # draw boundaries for each block
    for i in range(len(BLOCKS)):
        for col in range(BLOCKS[i][0] + 1):
            x = (col * block_info['widths'][i]) + boundaries[i][0]
            frame = cv2.line(frame,
                             (x, boundaries[i][2]),
                             (x, boundaries[i][3]),
                             COLOURS[i],
                             LINE_WIDTH)

        for row in range(BLOCKS[i][1] + 1):
            y = (row * block_info['heights'][i]) + boundaries[i][2]
            frame = cv2.line(frame,
                             (boundaries[i][0], y),
                             (boundaries[i][1], y),
                             COLOURS[i],
                             LINE_WIDTH)

    return frame


def encode(filename):
    """
    Transforms a video file into a sample spike file and saves the sample.
    """

    print("Encoding " + filename, end='\t')

    # keep in mind that openCV uses BGR!
    capture = cv2.VideoCapture(os.path.join(DIRECTORY, filename))

    # print debug information
    if TEST_RUN:
        show_debug_information(capture)

    # check data
    if capture.get(cv2.CAP_PROP_FRAME_COUNT) < 3:
        clean_up_and_exit(capture, "File must have more than two frames")

    # create output windows
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Pixel Spikes", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Fovea Spikes", cv2.WINDOW_AUTOSIZE)

    # read first frame
    success, frame_t0 = capture.read()
    if not success:
        clean_up_and_exit(capture, "Could not read first frame")

    # set initial focus coordinates to centre of image
    focus = (int(frame_t0.shape[1] / 2), int(frame_t0.shape[0] / 2))

    # calculate block size and position
    block_info = get_block_info(frame_t0.shape)

    # read second frame
    success, frame_t1 = capture.read()  # assume this one works if t0 worked

    # create output array
    sample = np.zeros((1, block_info['total_number'] + block_info['fovea_number']))

    while success:
        # convert frames to greyscale
        f0_grey = cv2.cvtColor(frame_t0, cv2.COLOR_BGR2GRAY)
        f1_grey = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)

        # calculate difference between pixel brightness
        pixel_spikes = get_frame_diff_as_spikes(f0_grey, f1_grey)

        # get current block boundaries based on current focus area
        boundaries = get_boundaries(frame_t0.shape, block_info, focus)

        # get spikes for blocks and update focus
        spikes, focus = get_block_spikes(pixel_spikes, block_info, boundaries, focus)

        # get colour difference for focus
        fovea_pixel_spikes = get_fovea_spikes(frame_t0, frame_t1, boundaries[-1])

        # summarise the spikes into their blocks
        fovea_spikes = get_block_fovea(fovea_pixel_spikes, block_info)

        # add fovea row to spike row
        spikes = np.append(spikes, fovea_spikes, axis=1)

        # add spike row to final output
        sample = np.append(sample, spikes, axis=0)

        # draw block boundaries
        frame_t0 = draw_boundaries(frame_t0, block_info, boundaries)

        # update display windows
        cv2.imshow("Original", frame_t0)
        cv2.imshow("Pixel Spikes", pixel_spikes * 255)
        cv2.imshow("Fovea Spikes", fovea_pixel_spikes * 255)

        # stop when Esc key is pressed
        if cv2.waitKey(1) == 27:
            break

        # slow down processing to enable observation
        if TEST_RUN:
            time.sleep(0.1)

        # move to next image
        frame_t0 = frame_t1
        success, frame_t1 = capture.read()

    # remove the first line that we filled with zeros to create the sample array
    sample = sample[1:, :]

    save_result_file(filename, sample)

    # some stats to optimise thresholds based on spike rate
    periphery_spike_rate = np.mean(sample[:, :block_info['total_number']+1])
    fovea_spike_rate = np.mean(sample[:, block_info['total_number']+1:])
    print(f'Periphery {periphery_spike_rate:.6f}   Fovea {fovea_spike_rate:.6f}')

    return (periphery_spike_rate, fovea_spike_rate)


def get_block_fovea(fovea_pixel_spikes, block_info):
    """
    Summarises the fovea spikes into blocks to reduce the number of inputs.
    Returns row of spikes from left to right then top to bottom.
    """

    # if we only have one fovea block, we want to keep all original pixels
    if BLOCKS[-1] == [1, 1]:
        return np.reshape(fovea_pixel_spikes, (1, -1))

    # create output row
    spikes = np.zeros((1, block_info['fovea_number']))

    # keep track of current item in spike row
    block_index = 0
    
    # set dynamic threshold
    threshold = 2 * np.mean(fovea_pixel_spikes)

    # easier access
    fovea_block_width = block_info['widths'][-1]
    fovea_block_height = block_info['heights'][-1]

    for col in range(BLOCKS[-1][0]):
        col_start = (col * fovea_block_width)
        col_end = col_start + fovea_block_width

        for row in range(BLOCKS[-1][1]):
            row_start = (row * fovea_block_height)
            row_end = row_start + fovea_block_height

            # calculate block's spike rate
            block_mean = np.mean(fovea_pixel_spikes[row_start:row_end, col_start:col_end])

            # check if we need to create a spike
            if block_mean > threshold:
                spikes[0, block_index] = 1

            # increase counter to keep track of next spike
            block_index += 1

    return spikes


def get_block_info(frame_shape):
    """
    Calculates the sizes of the blocks based on the size of the frame.
    """

    height = frame_shape[0]
    width = frame_shape[1]

    # calculate number of pixels in each block and total number of blocks
    block_widths = []
    block_heights = []
    total_number_of_blocks = 0
    for i in range(len(BLOCKS)):
        block_widths.append(int(width / ((BLOCK_SCALING_FACTOR ** i) * BLOCKS[i][0])))
        block_heights.append(int(height / ((BLOCK_SCALING_FACTOR ** i) * BLOCKS[i][1])))
        total_number_of_blocks += BLOCKS[i][0] * BLOCKS[i][1]

    # remove fovea block
    total_number_of_blocks -= BLOCKS[-1][0] * BLOCKS[-1][1]

    # build return dict
    block_info = {'widths': block_widths,
                  'heights': block_heights,
                  'total_number': total_number_of_blocks}

    # distinguish between pixel-wise and block-wise fovea
    a, b = 0, 0
    if BLOCKS[-1] == [1, 1]:
        a = block_heights[-1]
        b = block_widths[-1]
    else:
        a = BLOCKS[-1][0]
        b = BLOCKS[-1][1]

    block_info['fovea_number'] = a * b

    # debug info
    if TEST_RUN:
        print("Number of input channels required for periphery blocks: ",
              block_info['total_number'])
        print("Number of input channels required for fovea blocks: ",
              a, "x", b, "=", block_info['fovea_number'])
        m = block_heights[-1] * BLOCKS[-1][0]
        n = block_widths[-1] * BLOCKS[-1][1]
        print("Size of the fovea in pixels:", m, "x", n, "=", m*n)

    return block_info


def get_block_spikes(pixel_spikes, block_info, boundaries, focus):
    """
    Calculates which block emits a spike. Also determines most active region.
    Returns row of spikes going from outermost to innermost block and
    from left to right then from top to bottom.
    Returns coordinates of focus.
    """

    # create output row
    spikes = np.zeros((1, block_info['total_number']))

    # keep track of current item in spike row
    block_index = 0

    # keep track of most active region
    highest_spike_rate = 0.0
    
    # set dynamic threshold
    threshold = 2 * np.mean(pixel_spikes)

    for i in range(len(BLOCKS) - 1):
        for col in range(BLOCKS[i][0]):
            col_start = (col * block_info['widths'][i]) + boundaries[i][0]
            col_end = col_start + block_info['widths'][i]

            for row in range(BLOCKS[i][1]):
                row_start = (row * block_info['heights'][i]) + boundaries[i][2]
                row_end = row_start + block_info['heights'][i]

                # calculate block's spike rate
                block_mean = np.mean(pixel_spikes[row_start:row_end, col_start:col_end])

                # check if we need to create a spike
                if block_mean > threshold:
                    spikes[0, block_index] = 1

                # increase counter to keep track of next spike
                block_index += 1

                # find block with most activity
                if block_mean > highest_spike_rate:
                    highest_spike_rate = block_mean
                    # set actvity centre to centre of block
                    focus = (int((col_start + col_end) / 2),
                             int((row_start + row_end) / 2))

    return spikes, focus


def get_boundaries(frame_shape, block_info, focus):
    """
    Calculates the current outer boundaries of the blocks.
    The blocks must not go over the edges of the frame.
    """

    max_col = frame_shape[1]
    max_row = frame_shape[0]

    widths = block_info['widths']
    heights = block_info['heights']

    # calculate block position for drawing
    boundaries = []  # one entry per block level - start_col, end_col, start_row, end_row
    boundaries.append([0, max_col, 0, max_row])  # level 0 always covers full frame

    for i in range(1, len(BLOCKS)):
        start_col = int(focus[0] - ((BLOCKS[i][0] / 2) * widths[i]))
        if start_col < 0:
            start_col = 0

        end_col = start_col + (BLOCKS[i][0] * widths[i])
        if end_col > max_col:
            start_col -= (end_col - max_col)
            end_col = max_col

        start_row = int(focus[1] - ((BLOCKS[i][1] / 2) * heights[i]))
        if start_row < 0:
            start_row = 0

        end_row = start_row + (BLOCKS[i][1] * heights[i])
        if end_row > max_row:
            start_row -= (end_row - max_row)
            end_row = max_row

        boundaries.append([start_col, end_col, start_row, end_row])

    return boundaries


def get_fovea_spikes(frame_t0, frame_t1, fovea_boundaries):
    """
    Calculates if the colours of the pixels in the fovea block differ enough
    to create a spike. For this, it uses the CIEDE 2000 Delta E method.
    Returns a two-dimensional matrix of spikes.
    """

    # transform fovea area to CIELAB colour space
    # fovea boundaries are in order start_col, end_col, start_row, end_row
    fovea0_lab = cv2.cvtColor(frame_t0[fovea_boundaries[2]:fovea_boundaries[3],
                                       fovea_boundaries[0]:fovea_boundaries[1],
                                       :],
                              cv2.COLOR_BGR2Lab)
    fovea1_lab = cv2.cvtColor(frame_t1[fovea_boundaries[2]:fovea_boundaries[3],
                                       fovea_boundaries[0]:fovea_boundaries[1],
                                       :],
                              cv2.COLOR_BGR2Lab)

    # calculate delta E for each pixel
    delta_e_array = cd.delta_E_CIE2000(fovea0_lab, fovea1_lab)

    # threshold function will set all elements > fovea_threshold to 1
    spikes = cv2.threshold(delta_e_array, FOVEA_THRESHOLD, 1, cv2.THRESH_BINARY)[1]

    return spikes


def get_frame_diff_as_spikes(frame0, frame1):
    """
    Calculates difference between two frame arrays.
    frame0 is the first frame and frame1 is the frame following frame0.
    Returns a difference matrix that has the same shape as frame0 and frame1 -
    if they are in greyscale, they have rows and columns,
    and if they are in colour, they also have channels. (OpenCV uses BGR!)
    """

    # calculate absolute differences
    diff = cv2.absdiff(frame1, frame0)

    # threshold function will set all elements > pixel_threshold to 1
    pixel_spikes = cv2.threshold(diff, PIXEL_THRESHOLD, 1, cv2.THRESH_BINARY)[1]

    return pixel_spikes


def save_result_file(filename, sample):
    """
    Saves the sample that was incrementally created.
    The file name is equal the video name.
    """

    # save file as csv
    np.savetxt(os.path.join(DIRECTORY2, filename[:-4]) + '.csv',
               sample, fmt='%1.0f', delimiter=',', newline='\r\n')


def show_debug_information(capture):
    """
    Prints out some information about the current video.
    """

    print("Frames per second:", capture.get(cv2.CAP_PROP_FPS))
    print("Number of frames:", capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame width:", capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Frame height:", capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


###############################################################################
###                              PROGRAM LOGIC                              ###
###############################################################################

periphery_spike_rates = []
fovea_spike_rates = []

with os.scandir(DIRECTORY) as directory:
    for item in directory:
        if item.is_file() and item.name.endswith(FILETYPE):
            periphery_spike_rate, fovea_spike_rate = encode(item.name)
            periphery_spike_rates.append(periphery_spike_rate)
            fovea_spike_rates.append(fovea_spike_rate)

print("Mean periphery spike rate:", np.mean(periphery_spike_rates))
print("Mean fovea spike rate:", np.mean(fovea_spike_rates))

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.set_title("Fovea Spike Rates")
ax0.hist(fovea_spike_rates, bins=40)
ax1.set_title("Periphery Spike Rates")
ax1.hist(periphery_spike_rates, bins=40)
fig.tight_layout()
plt.show()

