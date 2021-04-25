function [ coord ] = get_input_coordinates(blocks)
%GET_INPUT_COORDINATES Returns a list of input coordinates for the blocks.

v1_data = load('v1_wang_xyz_ang_ecc.mat');
v1_data = v1_data.v1_wang_xyz_ang_ecc;

[cols, rows] = get_block_centres(blocks);
[ret_cols, ret_rows] = get_ang_ecc_pixel(v1_data(:,4), v1_data(:,5));
coord = zeros(size(cols,1), 3);

for i = 1:size(cols,1)
    [~, ind] = min(sqrt((ret_cols - cols(i)).^2 + (ret_rows - rows(i)).^2));
    coord(i,:) = v1_data(ind, 1:3);
end

coord = round(coord);

end
