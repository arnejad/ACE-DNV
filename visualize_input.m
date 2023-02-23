% This is mid-step to check the found information matched with frames

participantNum = 1;
activityNum = 2;


v = VideoReader('/media/ashdev/Expansion/data/GiW/1/1/world.mp4');

lbls = load("/media/ashdev/Expansion/data/GiW/res/lbls_p1_a1_l6.csv");
feats = load("/media/ashdev/Expansion/data/GiW/res/feats_p1_a1.csv");
gazes = load("/media/ashdev/Expansion/data/GiW/res/gazes_p1_a1.csv");
frames = load("/media/ashdev/Expansion/data/GiW/res/frames_p1_a1.csv");

patchSim = feats(:,5);
% gazes = feats(:,2:3);
gaze_rot = feats(:,1);
head_rot = feats(:,3);

% gazes = gazes .* [1920, 1080];


f = 1;
t = 1;

figure(1)

while (1)
    frame = readFrame(v);
%     imshow(frame)
    if f < frames(t) 
        f = f+1; fprintf(strcat(string(f), "\n"));
        continue
    end
    
    subplot(5,1,1)
    imshow(frame)

    %find next gazeindex
    hold on
    scatter(gazes(t,1), gazes(t,2), 50, 'MarkerFaceColor',[1 0 0])
    hold off
    
    
    subplot(5,1,2)
    plot(patchSim(1:t))
    title("patch similarity")

    subplot(5,1,3)
    plot(gaze_rot(1:t))
    title("gaze rotation")

    subplot(5,1,4)
    plot(head_rot(1:t))
    title("head rotation")

    subplot(5,1,5)
    heatmap(lbls(1:t)')
    title("labels")
    
    t = t+1;
    f = f+1;
    pause(0.1)
end
