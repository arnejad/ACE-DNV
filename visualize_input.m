% This is mid-step to check the found information matched with frames


v = VideoReader('/media/ashdev/Expansion/data/GiW/1/1/world.mp4');

lbls = load("/media/ashdev/Expansion/data/GiW/res/lbls.csv");
feats = load("/media/ashdev/Expansion/data/GiW/res/feats.csv");
gazes = load("/media/ashdev/Expansion/data/GiW/res/gazes.csv");
frames = load("/media/ashdev/Expansion/data/GiW/res/frames.csv");

patchSim = feats(:,5);
% gazes = feats(:,2:3);
gaze_rot = feats(:,1);
head_rot = feats(:,3);



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

    subplot(5,1,3)
    plot(gaze_rot(1:t))
    
    subplot(5,1,4)
    plot(head_rot(1:t))

    subplot(5,1,5)
    plot(lbls(1:t))
    
    t = t+1;
    f = f+1;
    pause(0.1)
end
