function [melSpecdb,gammaSpecdb]=bv_auditory_demo(file_name)

[audioIn,fs] = audioread(file_name);

numBands = 40;

spec = melSpectrogram(audioIn,fs, ...
    'Window',hamming(400,'periodic'), ...
    'OverlapLength',240, ...
    'FFTLength',1024, ...
    'NumBands',numBands);

melSpecdb=20*log10(spec+eps);

[D,F] = gammatonegram(audioIn,fs,0.025,0.010,numBands);
gammaSpecdb=20*log10(D+eps);

end
