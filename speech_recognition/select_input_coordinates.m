function [ input_coordinates, indices ] = select_input_coordinates( coordinates, number_of_inputs, by_value, plot_result )
% selects a number of evenly spaced points from a list
% if by_value is true, selects them uniformly by value, assuming they are sorted
% otherwise just selects uniformly from list

if (number_of_inputs < 1)
    error('Must select at least one input neuron!');
end

number_of_coordinates = size(coordinates, 1);

if (number_of_inputs == 1)
    % we just return the middle point
    indices = int16(number_of_coordinates / 2);    
end

if (number_of_inputs == 2)
    % we just return the first and last point
    indices = [1, number_of_coordinates];
end

if (number_of_inputs > 2)
    % stepwise selection
    if (by_value)
        % so we select uniformly based on values
        mini = min(coordinates(:, 4));
        maxi = max(coordinates(:, 4));
        values = mini:(maxi-mini) / (number_of_inputs - 1):maxi;
        indices = zeros(1, size(values, 2), 'int16');
        my_copy = coordinates(:, 4);
        for i = 1:size(values, 2)
            [~, idx] = min(abs(my_copy - values(i)));
            indices(i) = idx;
            my_copy(idx) = 10000;
        end
    else
        % so we select uniformly just on (sorted) index
        indices = int16(1:(number_of_coordinates-1) / (number_of_inputs - 1):number_of_coordinates);
    end
    % just making sure we have the correct number
    if (length(indices) < number_of_inputs)
        indices = [indices, number_of_coordinates];
    end
end

input_coordinates = coordinates(indices, :);

if (plot_result)
    figure('Color','w', 'NumberTitle', 'off', 'Name', 'Selected input coordinates');
    plot(coordinates(:, 4), 1:number_of_coordinates);
    hold on;
    scatter(input_coordinates(:, 4), indices,'filled')
    if (by_value)
        title('Selected indices by value');
    else
        title('Selected indices by index');
    end
    xlabel('Values')
    ylabel('Indices')
end

end
