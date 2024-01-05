
% This code will show the total distribution of the chosen samples


%drag and drop the files.txt file in the workspace to load it as a table


inp_dir = '/media/ashdev/Expansion/data/GiW';


files = readtable(fullfile(inp_dir,'files_all_lblr6.txt'));

activities = table2array(files(:,"Var3"));

lblr = 6;

total_hists = zeros(1, 5);

for i=1:size(files,1)


    filename = strcat("PrIdx_", num2str(table2array(files(i,"Var1"))),"_TrIdx_", num2str(table2array(files(i,"Var2"))),"_Lbr_", num2str(lblr) ,".mat");
    
    
    load(fullfile(inp_dir, 'Extracted_Data/',activities(i), "Labels", filename));


    temp_hist= histogram(LabelData.Labels, [-0.5, 0.5, 1.5,2.5, 3.5,4.5, 5.5]);
    
    total_hists = total_hists + temp_hist.Values(2:end);

end

X = categorical({'Fixation','Gaze Pursuit','Saccade','Blink', 'Gaze following'});

bar(X, total_hists)
ylim([0 400000])


