function [ polar_angle , eccentricity ] = get_pixel_ang_ecc(col, row, width, height)
%GET_PIXEL_ANG_ECC Returns the polar angle and eccentricity for a given
%pixel relative to the centre of a 176 x 100 pixel frame.

if nargin < 3
    width = 176;
    height = 100;
end

centre_col = width/2;
centre_row = height/2;

delta_col = centre_col - col;
delta_row = centre_row - row;

polar_angle = atand(delta_col./delta_row); % inverted tan in degrees
%                     0                0
%                  /     \          /     \
% convert from  +90      -90  to  90       270  to match the retinotopy data
%                  \     /          \     / 
%                     0               180
polar_angle(delta_row < 0) = polar_angle(delta_row < 0) + 180;
polar_angle(delta_row >= 0 & delta_col < 0) = polar_angle(delta_row >= 0 & delta_col < 0) + 360;

eccentricity = sqrt(delta_col.^2 + delta_row.^2); % euclidian distance
eccentricity = eccentricity * 16 / 200; % convert from pixel to degrees

end
