function [ cc ] = myCeps( x, p, fftLen )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
frame = x(1:fftLen, 1);
%p是梅尔频率倒谱系数 数值可变 20左右ok
%fftlen是一帧长度 
FR = fftshift(fft(fftshift(frame)));
%取正半轴
FR = FR(1025:end, 1);
%最高梅尔频率值 22050是因为采样率44100 傅里叶变换是-22050-22050
maxmelf = 2595*log10(1+22050/700);
sidewidth = maxmelf/(22+1);
%生成滤波器组
index = 0:21;
filterbankcenter = (10.^(((index+1)*sidewidth)/2595)-1)*700;
filterbankstart = (10.^((index*sidewidth)/2595)-1)*700;
filterbankend = (10.^(((index+2)*sidewidth)/2595)-1)*700;
filterbankcenter = floor(filterbankcenter*1024/22050);
filterbankstart = floor(filterbankstart*1024/22050);
filterbankend = floor(filterbankend*1024/22050);
filterbankstart(1) = 1;

filtmag = zeros(1024, 1);
tbfCoef = zeros(22, 1);
%降维 把每个滤波器率过后的值加起来 得到一个值然后一帧就是一个22维信号 第一个分量是跟音色有关，所以丢去。
for i = 1:22
    for j = filterbankstart(i):filterbankcenter(i)
        filtmag(j, 1) = (j-filterbankstart(i))/(filterbankcenter(i)-filterbankstart(i));
    end
    for j = filterbankcenter(i):filterbankend(i)
        filtmag(j, 1) = (filterbankend(i)-j)/(filterbankend(i)-filterbankcenter(i));
    end
	%滤过之后的spectragram
    tbfCoef(i, 1) = sum(FR(filterbankstart(i):filterbankend(i)).*filtmag(filterbankstart(i):filterbankend(i)));
end

%整体用dct做倒谱 因为速度快。
%转换成DB 在dtft之后直接对幅度的绝对值去log
tbfCoef = log(abs(tbfCoef));    
%再对log后的取余弦变化
cc = dct(tbfCoef);
cc = cc(1:p, 1);
end

