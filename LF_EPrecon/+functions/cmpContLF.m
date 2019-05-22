function contLF = cmpContLF(cam, scene)
    printf('Generating contLF from scene...');

    % Reorder Objects according to the Depths
    n = length(scene.arg.positions);
    
    objects = scene.arg.objects;
    depths = zeros(n, 1);
    xyOffsets = zeros(n, 2);
        
    for i = 1:n
        position = cell2mat(scene.arg.positions(i)); % Use cell2mat to convert the cells to a numeric array
        depths(i, 1) = position(1, 3);
        xyOffsets(i, :) = position(1:2);
    end
    
    [~, order] = sort(depths);
    objects = objects(order);
    depths = depths(order);
    xyOffsets = xyOffsets(order, :);
    
    % Compute the Continuous Lightfield with Linear Transformation and take Occlusion into Account    
    contLF = @(x, y, u, v) 0;
    window = @(x, y, u, v) 1;
    
    %D = max(cam.arg.focalStackDis);
    %D = cam.arg.focalStackDis(end);
    D = cam.arg.refDis;
    f = cam.arg.focalLen;
    for i = 1:n                                                             % Starts from near-lens object to far-lens object
        d = depths(i);
        object = objects{i};                                                % Content indexing with {} and cell indexing with ()
        xyOffset = xyOffsets(i, :);
        
        T = [ -d/D, (1 - d/f + d/D) ];

        objLF = @(x, y, u, v) object(T(1)*x+T(2)*u-xyOffset(1), T(1)*y+T(2)*v-xyOffset(2));% / (d/1e3)^2;
        occludedObjLF = @(x, y, u, v) objLF(x, y, u, v) .* window(x, y, u, v);
        window = @(x, y, u, v) window(x, y, u, v) .* (occludedObjLF(x, y, u, v) == 0);
        contLF = @(x, y, u, v) contLF(x, y, u, v) + occludedObjLF(x, y, u, v);                 
    end
end
