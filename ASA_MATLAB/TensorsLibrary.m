classdef TensorsLibrary
    %TENSORSLIBRARY Summary of this class goes here
    %   Detailed explanation goes here


    methods

        %% Get All Integer Partitions
        function sortedOutput = getAllIntegerPartitions(obj,number)
            % getAllIntegerPartitions Generates all unique integer partitions of a given number.
            %   sortedOutput = getAllIntegerPartitions(number)
            %
            %   Inputs:
            %       number (positive integer) - The target number to partition into sums of natural numbers.
            %
            %   Outputs:
            %       sortedOutput (cell array) - A cell array where each cell contains a unique,
            %                                   sorted combination of natural numbers that sum to 'number'.
            %
            %   Example:
            %       % Generate all unique integer partitions of the number 5
            %       partitions = getAllIntegerPartitions(5);
            %       celldisp(partitions);
            %
            %       % Expected Output:
            %       %
            %       %     {[1 4]}
            %       %     {[1 1 3]}
            %       %     {[2 3]}
            %       %     {[1 2 2]}
            %       %     {[1 1 1 2]}
            %       %     {[5]}

            output = {};    % Initialize empty cell array for output

            for i = 1:number-1  % Iterate through all natural numbers less than 'number'
                diff = number - i; % Calculate the difference

                if diff > 0 % If positive difference
                    sub_partitions = getAllIntegerPartitions(obj, i);  % Recursive call to get partitions for lower number
                    for j = 1:length(sub_partitions)
                        current_partition = sub_partitions{j};
                        new_partition = [current_partition, diff];
                        output = [output; {new_partition} ];  % Append new partition to output
                    end
                end
            end

            output = [output; {number}]; % Always add the number itself as a partition

            % Step 1: Sort each partition in ascending order
            for k = 1:length(output)
                output{k} = sort(output{k});
            end

            % Step 2: Remove duplicate partitions using MATLAB's 'unique' function
            partition_strings = cellfun(@(c) sprintf('%d_', c), output, 'UniformOutput', false);
            [~, unique_indices] = unique(partition_strings);
            unique_output = output(unique_indices);

            sortedOutput = unique_output;  % Update output with unique, sorted partitions
        end



        %% Tensor Subindices Combinations in Tensors Multiplication
        function unique_combinations = getTensorIndicesCombinations(obj, S, desired_order, tensor_orders)
            % getTensorIndicesCombinations Generates unique combinations of tensor indices based on specified orders.
            %   unique_combinations = getTensorIndicesCombinations(S, desired_order, tensor_orders)
            %
            %   Inputs:
            %       S (array of positive integers) - A set of subindices to be partitioned into tensors.
            %       desired_order (positive integer) - The total order that must equal the sum of tensor_orders.
            %       tensor_orders (array of positive integers) - Specifies the order of each tensor.
            %
            %   Outputs:
            %       unique_combinations (cell array) - A cell array where each row represents a unique
            %                                          combination of tensor indices adhering to the specified orders.
            %
            %   Example:
            %       % Define the set of subindices
            %       S = [1, 2, 3];
            %
            %       % Define the desired total order and tensor orders
            %       desired_order = 3;
            %       tensor_orders = [2, 1];
            %
            %       % Generate unique tensor index combinations
            %       combinations = getTensorIndicesCombinations(S, desired_order, tensor_orders);
            %       celldisp(combinations);
            %
            %       % Expected Output:
            %       %
            %       %     {'1 2'    '3'}
            %       %     {'1 3'    '2'}
            %       %     {'2 3'    '1'}

            % Validate
            if desired_order ~= sum(tensor_orders)
                error('Desired order must equal the sum of tensor orders.');
            end

            num_tensors = length(tensor_orders);

            % Generate all possible partitions of S into tensors with specified orders
            partitions = getIndicesPartitions(obj, S, tensor_orders);

            % Initialize cell array to hold unique combinations
            combinations_list = {};

            for k = 1:size(partitions, 1)
                partition = partitions(k, :); % Cell array of tensors' indices
                % For each tensor, sort its indices
                tensors_sorted = cellfun(@sort, partition, 'UniformOutput', false);
                % Convert tensors to strings
                tensors_strings = cellfun(@(x) char(x), tensors_sorted, 'UniformOutput', false);
                % To account for swapping tensors, sort the tensors lexicographically
                tensors_concatenated = sort(tensors_strings);
                % Combine the tensors into a single string for comparison
                combined_key = strjoin(tensors_concatenated, '|');
                % Check if this combination is already in the list
                if ~ismember(combined_key, combinations_list)
                    combinations_list{end+1} = combined_key;
                end
            end

            % Now, parse the combinations_list to create the output table
            num_combinations = length(combinations_list);
            unique_combinations = cell(num_combinations, num_tensors);
            for i = 1:num_combinations
                combination = combinations_list{i};
                tensors = strsplit(combination, '|');
                unique_combinations(i, :) = tensors;
            end
        end

        function partitions = getIndicesPartitions(obj, S, tensor_orders)
            % getIndicesPartitions Generates all possible partitions of indices into tensors based on specified orders.
            %   partitions = getIndicesPartitions(S, tensor_orders)
            %
            %   Inputs:
            %       S (array of positive integers) - A set of subindices to be partitioned.
            %       tensor_orders (array of positive integers) - Specifies the order (size) of each tensor.
            %
            %   Outputs:
            %       partitions (cell array) - A cell array where each row contains the indices assigned
            %                                 to each tensor based on the specified tensor_orders.
            %
            %   Example:
            %       % Define the set of subindices
            %       S = [1, 2, 3];
            %
            %       % Define the tensor orders
            %       tensor_orders = [2, 1];
            %
            %       % Generate all possible index partitions
            %       partitions = getIndicesPartitions(S, tensor_orders);
            %       celldisp(partitions);
            %
            %       % Expected Output:
            %       %
            %       %     {[1 2], [3]}
            %       %     {[1 3], [2]}
            %       %     {[2 3], [1]}

            num_tensors = length(tensor_orders);
            num_Subindices = length(S);
            partitions = {};

            % Base case: if there is only one tensor, the partition is S itself if sizes match
            if num_tensors == 1
                if num_Subindices == tensor_orders(1)
                    partitions = {S};
                else
                    partitions = {};
                end
                return;
            end

            % For the first tensor, generate all combinations of its indices
            first_order = tensor_orders(1);
            combinations_first = nchoosek(S, first_order);
            num_combinations_first = size(combinations_first, 1);

            for i = 1:num_combinations_first % For each of the indices combinations of the first tensor
                first_tensor_indices = combinations_first(i, :);
                remaining_indices = setdiff(S, first_tensor_indices, 'stable'); % Get remaining indices to be covered

                % Recursively generate partitions for the remaining tensors and indices
                sub_partitions = getIndicesPartitions(obj,remaining_indices, tensor_orders(2:end));

                % Combine the first tensor indices with the sub-partitions
                if ~isempty(sub_partitions)
                    num_sub_partitions = size(sub_partitions, 1);
                    for j = 1:num_sub_partitions
                        % Concatenate the indices
                        partition = [{first_tensor_indices}, sub_partitions(j, :)];
                        partitions = [partitions; partition];
                    end
                end
            end
        end


    end
end

