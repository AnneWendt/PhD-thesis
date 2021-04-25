% resample the spoken words corpus from 8kHz to 100kHz for cochlea.py

source_folder = 'randomly_selected_samples\';
target_folder = 'upsampled\';

source_fs = 16000;
target_fs = 100000;

% two values that are needed for MATLAB's resample function
[p,q] = rat(target_fs/source_fs);

% get all the filenames
info = dir(source_folder);
info = info(3:end); %the first two are . and .. pointing to itself and the parent directory

for file_id = 1:length(info)

    filename = [source_folder info(file_id).name];
    [sound, fs] = audioread(filename);

    assert(fs==source_fs);

    new_sound = resample(sound,p,q);

    new_filename = [target_folder info(file_id).name(1:end-3) 'wav'];
    audiowrite(new_filename, new_sound, target_fs);

end
