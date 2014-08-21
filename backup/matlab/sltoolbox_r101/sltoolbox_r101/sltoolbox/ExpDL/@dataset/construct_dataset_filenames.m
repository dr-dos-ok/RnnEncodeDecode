function ds = construct_dataset_fns(ds, name, format, filenames, labels)
%CONSTRUCT_DATASET_FNS Constructs Dataset object from filenames
%
% $ Syntax $
%   - ds = construct_dataset_fns(ds, name, format, filenames, labels)
%
% $ Arguments $
%   - name:             the name of the dataset
%   - format:           the format of file samples
%   - filenames:        the filenames of the image samples
%   - labels:           the labels of the corresponding samples
%   - ds:               the constructed object
%
% $ Description $
%   - ds = construct_dataset_fns(name, format, filenames, labels) 
%     constructs a dataset object from filenames and corresponding labels 
%     identifying the class of the samples.
%
% $ Remarks $
%   - The dataset object will be constructed as sample set.
%
% $ History $
%   - Created by Dahua Lin on Jul 26th, 2005
%

%% parse and verify input arguments

N = length(filenames);
if length(labels) ~= N
    error('The length of labels is not consistent with that of filenames');
end

%% construct

ds.version = '1.00';
ds.name = name;
ds.unittype = 'Sample';
ds.format = format;
ds.author = 'unknown';
ds.description = 'generated by construct_dataset_fns.m';
ds.attribs = [];

ds.units = [];

for i = 1 : N;
    ds.units(i).class_id = labels(i);
    ds.units(i).filename = filenames{i};
    ds.units(i).attribs = [];
end





