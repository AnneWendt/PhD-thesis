import matplotlib.pyplot as plt
import numpy as np
import os


def fill_array_centred(array, rows):
    front_rows = int(rows / 2)
    back_rows = rows - front_rows # in case of odd number
    cols = np.shape(array)[1]
    
    return np.concatenate((np.zeros((front_rows, cols), dtype=int), array, np.zeros((back_rows, cols), dtype=int)), axis=0)


def fill_array_front(array, rows):
    cols = np.shape(array)[1]
    
    return np.concatenate((array, np.zeros((rows, cols), dtype=int)), axis=0)


def generate_list_of_filenames():
    filenames = []

    for sam in range(1, 251):
        word = ''
    
        if sam < 51:
            word = 'bird'
        elif sam < 101:
            word = 'down'
        elif sam < 151:
            word = 'stop'
        elif sam < 201:
            word = 'tree'
        else:
            word = 'up'
        
        filename = 'sam' + str(sam) + '_' + word + '.csv'
        filenames.append(filename)
    
    return filenames


def remove_leading_rows_of_zeros(array):
    while True:
        # break if no rows left
        if np.shape(array)[0] == 0:
            return array
        
        # break if row with content is detected
        # but only if next row also has content
        if array[0,:].sum() > 0 and array[1,:].sum() > 0:
            return array
        
        # keep removing rows until either of the above conditions is met
        array = np.delete(array, 0, axis=0)


def remove_trailing_rows_of_zeros(array):
    while True:
        # break if no rows left
        if np.shape(array)[0] == 0:
            return array
        
        # break if row with content is detected
        # but only if next row also has content
        if array[-1,:].sum() > 0 and array[-2,:].sum() > 0:
            return array
        
        # keep removing rows until either of the above conditions is met
        array = np.delete(array, -1, axis=0)


def main(template_name):
    print('Combining', template_name)

    audio_src = os.path.join('speech', 'encoded_samples', template_name)
    video_scr = os.path.join('videos', 'encoded_samples', template_name)
    tar_dir = os.path.join('combined', template_name)
    
    filenames = generate_list_of_filenames()

    # some analysis
    audio_rows_before = []
    audio_rows_after = []
    video_rows_before = []
    video_rows_after = []

    # merge files from different folders
    for filename in filenames:
        audio_file = np.loadtxt(os.path.join(audio_src, filename), dtype='int', delimiter=',')
        video_file = np.loadtxt(os.path.join(video_scr, filename), dtype='int', delimiter=',')
        
        audio_rows_before.append(np.shape(audio_file)[0])
        video_rows_before.append(np.shape(video_file)[0])

        # remove rows of zeros
        audio_file = remove_leading_rows_of_zeros(audio_file)
        audio_file = remove_trailing_rows_of_zeros(audio_file)
        video_file = remove_leading_rows_of_zeros(video_file)
        video_file = remove_trailing_rows_of_zeros(video_file)

        # adjust video file length based on biological observations
        video_file = np.repeat(video_file, 3, axis=0)
        
        audio_rows_after.append(np.shape(audio_file)[0])
        video_rows_after.append(np.shape(video_file)[0])

        # calculate how many rows need to be padded
        row_diff = np.shape(audio_file)[0] - np.shape(video_file)[0]

        # if video file is longer than audio file, fill audio file
        if row_diff < 0:
            audio_file = fill_array_front(audio_file, abs(row_diff))
            
        # if audio file is longer than video file, fill video file
        if row_diff > 0:
            video_file = fill_array_front(video_file, row_diff)

        # join files and save as one
        combined = np.concatenate((audio_file, video_file), axis=1)
        #np.savetxt(os.path.join(tar_dir, filename), combined, fmt='%d', delimiter=',')

    # some statistics
    print("Audio file length ranged from", np.min(audio_rows_before), "to", np.max(audio_rows_before), "with mean", np.mean(audio_rows_before))
    print("Audio file length now ranges from", np.min(audio_rows_after), "to", np.max(audio_rows_after), "with mean", np.mean(audio_rows_after))
    print("Video file length ranged from", np.min(video_rows_before), "to", np.max(video_rows_before), "with mean", np.mean(video_rows_before))
    print("Video file length now ranges from", np.min(video_rows_after), "to", np.max(video_rows_after), "with mean", np.mean(video_rows_after))

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.set_title("Audio file length before")
    ax0.hist(audio_rows_before, bins=40)
    ax1.set_title("Audio file length after")
    ax1.hist(audio_rows_after, bins=40)
    fig.tight_layout()
    plt.show()

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.set_title("Video file length before")
    ax0.hist(video_rows_before, bins=40)
    ax1.set_title("Video file length after")
    ax1.hist(video_rows_after, bins=40)
    fig.tight_layout()
    plt.show()


main('MNI_by_4')
main('TAL_by_8')
