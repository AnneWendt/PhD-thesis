% resample the spoken words corpus from 8kHz to 100kHz for cochlea.py

source_folder = 'H:\Data\Anne\projects\Giovanni\my_experiment\benchmarking\free-spoken-digit-dataset-master\recordings\';
target_folder = 'H:\Data\Anne\projects\Giovanni\my_experiment\benchmarking\free-spoken-digit-dataset-master\upsampled\';

source_fs = 8000;
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

    filename = [target_folder info(file_id).name];
    audiowrite(filename,new_sound,target_fs);

end
