% this script shows the labeler- and activity-specific distribution of labels


inp_dir = '/media/ashdev/Expansion/data/GiW/Extracted_Data';

num_of_participants = 22;
num_of_lblrs = 6;
num_of_activities = 4;

% p_list = dir(inp_dir); p_list(1:2)=[]; p_list(end-1:end) = [];

activities = ["Ball_Catch", "Indoor_Walk", "Tea_Making", "Visual_Search"];



for a=1:num_of_activities
    
    total_hists = zeros(num_of_participants, 5);
    
    for lblr=1:num_of_lblrs

        lbl_hist = zeros(num_of_participants, 5);
        lbl_list = dir(fullfile(inp_dir, activities(a), 'Labels'));  
        lbl_list(1:2)=[];
        lbl_list = struct2cell(lbl_list);
        lbl_list  = lbl_list(1,:);
    
        lbl_list = lbl_list(contains(lbl_list, strcat('Lbr_', num2str(lblr))));
        
        for lbl=1:length(lbl_list)
            load(fullfile(inp_dir, activities(a), "Labels", lbl_list(lbl)));

            temp_hist= histogram(LabelData.Labels, [-0.5, 0.5, 1.5,2.5, 3.5,4.5, 5.5]);
            participant_num = split(erase(lbl_list(lbl), "PrIdx_"), '_');
            participant_num = str2num(cell2mat(participant_num(1)));

            lbl_hist(participant_num, :) = temp_hist.Values(2:end);
            
            total_hists(participant_num, :) = total_hists(participant_num, :) + temp_hist.Values(2:end);
            
        end
        hm = heatmap(lbl_hist);
        name = strcat(activities(a), " Labeler ", num2str(lblr));
        title(name)
        saveas(hm,fullfile("~/Desktop/heatmaps", strcat(name, '.jpg')));

    end

    hm = heatmap(total_hists);
    name = strcat(activities(a), " All Labelers prt-specific");
    title(name)
    saveas(hm,fullfile("~/Desktop/heatmaps", strcat(name, '.jpg')));

    
    X = categorical({'Fixation','Gaze Pursuit','Saccade','Blink', 'Gaze following'});
    hist_all= bar(X, sum(total_hists, 1));
    name = strcat(activities(a), " All Labelers all-prts");
    title(name)
    
    saveas(hist_all,fullfile("~/Desktop/heatmaps", strcat(name, '.jpg')));

end