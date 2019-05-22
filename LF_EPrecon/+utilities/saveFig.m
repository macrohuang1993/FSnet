function saveFig(figs, names, dirName)
    import utilities.genFile;

    if  numel(figs) ~= numel(names)
        error('Number of figures and names do not match');
    end
    
    for i = 1:numel(figs)
        file = genFile( [dirName names{i} '.jpg'] );
        saveas(figs(i), file, 'jpeg');
    end
end