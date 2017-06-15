% Code used to annotate test sequence video with predicted labels

clear
close all
clc

test = 2;
trials = 21;

for i = 1:trials
    % read predicted labels
    fid = fopen(['Tennis Data\Tests\daniel_lau_' num2str(test) '\Video\video_predictions_' num2str(test) '_' num2str(i) '.csv']);
    predictions = textscan(fid, '%s');
    fclose(fid);
    num_pred = length(predictions{1});
    
    if i > 1
        clear predict_test
    end
    time = [];
    for j = 1:num_pred
        splitit = strsplit(predictions{1}{j}, ',');
        time(j) = str2num(splitit{1});
        predict_test(j) = string(splitit{2});
    end
    stride = time(2) - time(1);

    % get corresponding video of game and count frames
    v = VideoReader(['Tennis Data\Tests\daniel_lau_' num2str(test) '\Video\editted_tennis_test_vid_' num2str(test) '.mp4']);
    num_frames = 0;
    while hasFrame(v)
        video = readFrame(v);
        num_frames = num_frames + 1;
    end

    % calculate for how many frames a predicted label applies
    label_length = stride*v.Duration/(stride*num_pred);
    frame_length = v.Duration/num_frames;
    stay_for_frames = round(label_length/frame_length);

    % loop through all frames and annotate with appropriate predicted labels
    counter = 0;
    v = VideoReader(['Tennis Data\Tests\daniel_lau_' num2str(test) '\Video\editted_tennis_test_vid_' num2str(test) '.mp4']);
    while hasFrame(v)
        counter = counter + 1;
        this_pred = ceil(counter/stay_for_frames);
        if this_pred > num_pred
            break
        end

        video = readFrame(v);

        txt = char(predict_test(this_pred));
        if txt(1) == 'o'
            txt = 'overhand serve';
        end

        imshow(video), hold on
        text(25,80,txt,'Color','w','FontWeight','Bold','FontSize',40);

        this_frame = getframe;
        frames(counter) = im2frame(this_frame.cdata);
        hold off
    end

    % save video
    % Compression is recommended as size of created video will be large
    new_vid = VideoWriter(['annotated_' num2str(test) '_' num2str(i) '.mp4']);
    open(new_vid)
    writeVideo(new_vid, frames)
    close(new_vid)
end
