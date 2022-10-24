% This is mid-step to check the found information matched with frames


v = VideoReader('/home/ashdev/data/GiW/1/1/world.mp4');

lbls = load("~/data/GiW/res/lbls.csv");
feats = load("~/data/GiW/res/feats.csv");
frames = load("~/data/GiW/res/frames.csv");

patchSim = feats(:,1);
gazes = feats(:,2:3);
rot = feats(:,4);


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
    
    subplot(3,1,1)
    imshow(frame)

    %find next gazeindex
    hold on
    scatter(gazes(t,1), gazes(t,2), 50, 'MarkerFaceColor',[1 0 0])
    hold off
    
    
    subplot(3,1,2)
    plot(patchSim(1:t))
    
    subplot(3,1,3)
    plot(rot(1:t))
    
    t = t+1;
    f = f+1;
    pause(0.001)
end
