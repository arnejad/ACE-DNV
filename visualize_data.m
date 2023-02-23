% This is mid-step to check the found information matched with frames

participantNum = '1';
activityNum = '1';
activityName = "Indoor_Walk";


v = VideoReader(strcat('/media/ashdev/Expansion/data/GiW/',participantNum,'/', activityNum,'/world.mp4'));

load(strcat("/media/ashdev/Expansion/data/GiW/Extracted_Data/",activityName,"/Labels","/PrIdx_",participantNum,"_TrIdx_", activityNum ,"_Lbr_6.mat"));
load(strcat("/media/ashdev/Expansion/data/GiW/Extracted_Data/",activityName,"/ProcessData","/PrIdx_",participantNum,"_TrIdx_", activityNum ,".mat"));
% gazes = load(strcat("/media/ashdev/Expansion/data/GiW/Extracted_Data/gazes_p",participantNum,"_a", activityNum ,".csv"));
% frames = load(strcat("/media/ashdev/Expansion/data/GiW/Extracted_Data/frames_p",participantNum,"_a", activityNum ,".csv"));

gazes = ProcessData.ETG.POR;
% patchSim = feats(:,5);
% % gazes = feats(:,2:3);
% gaze_rot = feats(:,1);
% head_rot = feats(:,3);

% gazes = gazes .* [1920, 1080];


f = 1;
t = 1;

figure(1)

% 
subplot(3,1,1)
plot(gazes(27000:27200,1))
title("gaze x")

subplot(3,1,2)
plot(gazes(27000:27200,2))
title("gaze y")

subplot(3,1,3)
heatmap(LabelData.Labels(27000:27200)')
title("labels")


while (1)
    
    if t <= 22600
        t = t+1; fprintf(strcat(string(f), "\n"));
        continue
    end
    
    subplot(3,1,1)
    plot(gazes(22600:t,1))
    title("gaze x")

    subplot(3,1,2)
    plot(gazes(22600:t,2))
    title("gaze y")

    subplot(3,1,3)
    heatmap(LabelData.Labels(22600:t)', 'Colormap',summer)
    title("labels")
    
    t = t+1;
    f = f+1;
    pause(0.1)
end
