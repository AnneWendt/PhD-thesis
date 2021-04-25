function [ col , row ] = get_ang_ecc_pixel(polar_angle, eccentricity, width, height)
%GET_ANG_ECC_PIXEL Returns the pixel coordinates for a given polar angle
%and eccentricity relative to the centre of a pixel frame.

% set default values for frame size
if nargin < 3
    width = 176;
    height = 100;
end

% get central point
centre_col = width/2;
centre_row = height/2;

% convert to pixels
eccentricity = eccentricity * 100 / 8;

% calculate adjacent and opposite
delta_row = eccentricity .* cosd(polar_angle);
row = round(centre_row - delta_row);

delta_col = eccentricity .* sind(polar_angle);
col = round(centre_col - delta_col);

end
