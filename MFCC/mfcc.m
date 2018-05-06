file1 = dir('NoteSamples\BbClar\*.wav');
lb = length(file1);
windowham = hamming(2048);
Bpoint = zeros(length(file1), 3);
for i = 1:length(file1)
    temp = wavread(['NoteSamples\BbClar\',file1(i).name]);
    a = 1;
    b = 2048;
    for j = 1:5
        frame = temp(a + (10+j-1-1)*2048: b + (10+j-1-1)*2048, 1);
        frame = frame.*windowham;
        if ((i == 1) && (j == 1))
            Bb = frame;
        else
            Bb = [Bb frame];
        end
        C = myCeps(frame, 21, 2048);
        if ((i == 1) && (j == 1))
            BbCldat = C;
        else 
            BbCldat = [BbCldat  C];
        end
    end
end

save BbCldat.mat BbCldat -ascii;

file2 = dir('NoteSamples\Flute\*.wav');
lf = length(file2);
Fpoint = zeros(length(file2), 3);
for i = 1:length(file2)
    temp = wavread(['NoteSamples\Flute\',file2(i).name]);
    a = 1;
    b = 2048;
    for j = 1:5
        frame = temp(a + (10+j-1-1)*2048: b + (10+j-1-1)*2048, 1);
        frame = frame.*windowham;
        if ((i == 1) && (j == 1))
            Fl = frame;
        else
            Fl = [Fl frame];
        end
        C = myCeps(frame, 21, 2048);
        if ((i == 1) && (j == 1))
            Flutedat = C;
        else 
            Flutedat = [Flutedat  C];
        end
    end
end

save Flutedat.mat Flutedat -ascii;

file3 = dir('NoteSamples\Trumpet\*.wav');
lt = length(file3);
Tpoint = zeros(length(file3), 3);
for i = 1:length(file3)
    temp = wavread(['NoteSamples\Trumpet\',file3(i).name]);
    a = 1;
    b = 2048;
    for j = 1:5
        frame = temp(a + (10+j-1-1)*2048: b + (10+j-1-1)*2048, 1);
        frame = frame.*windowham;
        if ((i == 1) && (j == 1))
            Tp = frame;
        else
            Tp = [Tp frame];
        end
        C = myCeps(frame, 21, 2048);
        if ((i == 1) && (j == 1))
            Trumpetdat = C;
        else 
            Trumpetdat = [Trumpetdat  C];
        end
    end
end

save Trumpetdat.mat Trumpetdat -ascii;