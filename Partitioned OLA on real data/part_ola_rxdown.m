%T = table2array(rxdownconvout)

fs = 44100;
Ts = 1/fs;


dsamp = 2;
Tb = 20e-3;
samps = Tb*fs


n = 10;
delta_f = n/(2*Tb)


f1 = 800;
f0 = 200;


bits = [1,1,1,0,0,0,1,0,0,1,0];
tx_l = fsk(bits, samps, f0, f1, fs);
rx_l = pollute(bits, 0, 0, samps, f0, f1, fs, 0.8, 0*Ts, 10000);
tx = downsample(tx_l, dsamp);
rx = downsample(tx_l, dsamp); 

rtr = xcorr(rx, tx);
figure('Name', 'Cross-correlation')
plot(rtr)

rx_real = T(1:12288)
figure('Name', 'Real Received Message')
plot(rx_real)

rtr_real = xcorr(rx_real, tx)
figure('Name', 'Real Cross-correlation')
plot(rtr_real)

rtr_calc = T(12289:25600);
figure('Name', 'Real Cross-correlation')
plot(rtr_calc)

% plt.stem(rtr, linefmt="g:", markerfmt = 'g.', use_line_collection= True) 
% plt.show() 
% 
% plt.plot(rtr[882:882*3])
% plt.show()
% 
% mf_coef= np.flip(tx3_l)
% mf_fft = np.abs(np.fft.fft(mf_coef))
% plt.stem(mf_fft, linefmt="y:", markerfmt = 'y.', use_line_collection= True)
% plt.show()

function mn = fsk(bitseq, spb, f_0, f_1, F)
    T = 1/F;
    nb = size(bitseq, 2);
    num = 1:spb;
    mn = zeros(1, spb*nb);
    mn_len = 1:size(mn,2);
    
    for i = 1:nb
        if (bitseq(i) == 1) 
            bit = sin(2*f_1*pi*num*T);
        end
        if (bitseq(i) == 0)
            bit = sin(2*f_0*pi*num*T);
        end
        mn((i-1)*spb+1:i*spb) = bit;
    end
 
    figure('Name', 'Modulated Signal')
    plot(mn)
end


function rn = pollute(bitseq, len_f, len_b, spb, f_0, f_1, F, gain, timeshift, noise_snr)   
    
    T = 1/F;  
    nb = size(bitseq, 2);
    num = 1:spb;
    rn = zeros(1, spb*nb);
    phase1 = 2*pi*f_1*timeshift;
    phase0 = 2*pi*f_0*timeshift;

    for i = 1:nb
        if (bitseq(i) == 1) 
            bit = sin(2*f_1*pi*num*T + phase1);
        end
        if (bitseq(i) == 0)
            bit = sin(2*f_0*pi*num*T + phase0);
        end
        rn((i-1)*spb+1:i*spb) = bit;
    end

    rn = [zeros(1, len_f) rn zeros(1, len_b)];

    rn = awgn(rn , noise_snr);
    
    rn = gain*rn;
   
    figure('Name', 'Received Signal')
    plot(rn) 
end
