function install_paths()
%THE SCRIPT to install the sltoolbox paths to matlab system

fp = mfilename('fullpath');
rootdir = fileparts(fp);

subfolders = { ...
    'ann'; ...
    'cluster'; ...
    'core'; ...
    'discrete'; ...
    'ExpDL'; ...
    'fileio'; ...
    'graph'; ...
    'imgproc'; ...
    'interp'; ...
    'kernel'; ...
    'learn'; ...
    'manifold'; ...
    'perfeval'; ...
    'regression'; ...
    'smallmat'; ...
    'stat'; ...
    'subspace'; ...
    'subspace_ex'; ...
    'tensor'; ...
    'text'; ...
    'utils'; ...
    'utils_ex'; ...
    'visualize'; ...
    'xmlkits'};

n = length(subfolders);
folderpaths = cell(1, n);

for i = 1 : n
    folderpaths{i} = fullfile(rootdir, subfolders{i});
    fprintf('Add path: %s\n', folderpaths{i});    
end

addpath(folderpaths{:});
savepath;

disp('All paths have been added.');
disp(' ');  % a blank line



