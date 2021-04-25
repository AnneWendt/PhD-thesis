function [ cols , rows ] = get_block_centres(blocks, height, width)
%GET_BLOCK_CENTRES Returns the coordinates of the centres of the blocks
%defined in a 176 x 100 pixel frame.

if nargin < 3
    width = 176;
    height = 100;
end

focus_w = round(width/2);
focus_h = round(height/2);

block_scaling_factor = 4 ^ (1 / (size(blocks,1) - 1));

cols = [];
rows = [];

for i = 1:size(blocks,1)
    n_block_cols = blocks(i, 1);
    n_block_rows = blocks(i, 2);
    
    % get size of blocks
    block_width = round(width / ((block_scaling_factor ^ i-1) * n_block_cols));
    block_height = round(height / ((block_scaling_factor ^ i-1) * n_block_rows));
    
    % get starting point for block
    start_col = focus_w - round(((n_block_cols + 1) * block_width) / 2);
    start_row = focus_h - round(((n_block_rows + 1) * block_height) / 2);
    
    % arrangement of blocks is
    % 1 3
    % 2 4
    
    col_indices = [];
    for c = 1:n_block_cols
        col_indices = [col_indices; repmat(start_col + (c * block_width), n_block_rows, 1)];
    end
    cols = [cols; col_indices];
    
    row_indices = [];
    for r = 1:n_block_rows
        row_indices = [row_indices; start_row + (r * block_height)];
    end
    rows = [rows; repmat(row_indices, n_block_cols, 1)];
end


end
