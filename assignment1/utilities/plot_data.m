% x =[1	8940;
% 2	4763;
% 4	2536;
% 8	1411;
% % 2048	379;
% ];
% 
% hold on
% figure(1);
% title("Effect of Number of Threads on CPU Time");
% xlabel("Number of Threads");
% ylabel("CPU Time (ms)");
% 
% plot(x(:,1), x(:,2));
% hold off;


% 0.993
% 1.864
% 3.501
% 6.292
% 23.42




speedup = 0.993;
N = 1;

speedup = 1.864;
N = 2;

speedup = 3.501;
N = 4;

speedup = 6.292;
N = 8;

speedup = 23.42;
N = 2048;

P_estimated = ((1/speedup)-1) / ((1/N)-1)


