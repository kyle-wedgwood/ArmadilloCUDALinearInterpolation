% Test restriction based on CUDA code

% % Load data
firingInd  = load('testFiringInd.dat');
firingTime = load('testFiringTime.dat');
spikeInd   = load('testSpikeInd.dat');
spikeCount = load('testSpikeCount.dat');

noReal = 1000;
noThreads = 2^10;
noSpikes = 3;
noStoredSpikes = 900;

L = 3.0;

%% Do computation based on my initial calculation
noTotalSpikes = spikeCount(1);
xBar   = zeros(noSpikes,1);
tBar   = zeros(noSpikes,1);
tSqBar = zeros(noSpikes,1);
xtBar  = zeros(noSpikes,1);
count  = zeros(noSpikes,1);

for spikeNo = 0:noSpikes-1

  count(spikeNo+1)  = sum(spikeInd(1:noTotalSpikes)==spikeNo);
  xBar(spikeNo+1)   = sum(firingInd(1:noTotalSpikes).*(spikeInd(1:noTotalSpikes)==spikeNo));
  tBar(spikeNo+1)   = sum(firingTime(1:noTotalSpikes).*(spikeInd(1:noTotalSpikes)==spikeNo));
  xtBar(spikeNo+1)  = sum((firingTime(1:noTotalSpikes).*firingInd(1:noTotalSpikes)).*(spikeInd(1:noTotalSpikes)==spikeNo));
  tSqBar(spikeNo+1) = sum((firingTime(1:noTotalSpikes).*firingTime(1:noTotalSpikes)).*(spikeInd(1:noTotalSpikes)==spikeNo));

end

speedNum   = 0.0;
speedDenom = 0.0;
speedSeparate = zeros(noSpikes,1);

for spikeNo = 1:noSpikes

  speedNum   = speedNum   + xtBar(spikeNo) - 1/count(spikeNo)*xBar(spikeNo)*tBar(spikeNo);
  speedDenom = speedDenom + tSqBar(spikeNo)- 1/count(spikeNo)*tBar(spikeNo)*tBar(spikeNo);
  speedSeparate(spikeNo) = (xtBar(spikeNo)-1/count(spikeNo)*xBar(spikeNo)*tBar(spikeNo))/...
                (tSqBar(spikeNo)-1/count(spikeNo)*tBar(spikeNo)*tBar(spikeNo));
end

speedLoop = speedNum/speedDenom;
speedOne = sum(xtBar - 1./count.*xBar.*tBar)/sum(tSqBar - 1./count.*tBar.*tBar);

offset = zeros(noSpikes,1);
offsetSeparate = zeros(noSpikes,1);

for spikeNo = 1:noSpikes
  offset(spikeNo) = 1/count(spikeNo)*(xBar(spikeNo) - speedLoop*tBar(spikeNo));
  offsetSeparate(spikeNo) = 1/count(spikeNo)*(xBar(spikeNo) - ...
                            speedSeparate(spikeNo)*tBar(spikeNo));
end

% Plot results
figure(1);
hold on;
plot(firingTime(1:noTotalSpikes),firingInd(1:noTotalSpikes),'.');
x = sort(firingTime);

for spikeNo = 1:noSpikes
  y = speedLoop*x+offset(spikeNo);
  z = speedSeparate(spikeNo)*x+offsetSeparate(spikeNo);
  plot(x,y,x,z);
end

% Storage
averages = zeros(5*noSpikes*noReal,1);

%% New code
for realNo = 1:noReal

  count = spikeCount(realNo);

  for spikeNo = 0:noSpikes-1

    val.t   = 0.0;
    val.x   = 0.0;
    val.tSq = 0.0;
    val.xt  = 0.0;
    val.count = 0;

    for i = 1:count

      t = firingTime(i)*(spikeInd(i)==spikeNo);
      x = firingInd(i)*(spikeInd(i)==spikeNo);

      val.t = val.t+t;
      val.x = val.x+x;
      val.count = val.count+(spikeInd(i)==spikeNo);
      val.tSq = val.tSq+t*t;
      val.xt  = val.xt+t*x;

    end

    averages(0*noSpikes*noReal+spikeNo*noReal+realNo) = val.x;
    averages(1*noSpikes*noReal+spikeNo*noReal+realNo) = val.t;
    averages(2*noSpikes*noReal+spikeNo*noReal+realNo) = val.tSq;
    averages(3*noSpikes*noReal+spikeNo*noReal+realNo) = val.xt;
    averages(4*noSpikes*noReal+spikeNo*noReal+realNo) = val.count;

  end

end

%% Second step
U = zeros(noSpikes*noReal,1);

for realNo = 1:noReal

  xBar = zeros(noSpikes,1);
  tBar = zeros(noSpikes,1);
  noCross = zeros(noSpikes,1);
  speedNum   = 0.0;
  speedDenom = 0.0;
  offset = zeros(noSpikes,1);

  for spikeNo = 1:noSpikes

    noCross(spikeNo) = averages(4*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    xBar(spikeNo) = averages(0*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    tBar(spikeNo) = averages(1*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    tBarSq        = averages(2*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    xtBar         = averages(3*noSpikes*noReal+(spikeNo-1)*noReal+realNo);

    speedNum    = speedNum+(xtBar-1/noCross(spikeNo)*xBar(spikeNo)*tBar(spikeNo));
    speedDenom  = speedDenom+(tBarSq-1/noCross(spikeNo)*tBar(spikeNo)*tBar(spikeNo));

  end

  speed = speedNum/speedDenom;

  for spikeNo = 1:noSpikes
    offset(spikeNo) = 1/noCross(spikeNo)*(xBar(spikeNo)-speed*tBar(spikeNo));
  end

  U(realNo) = speed*2.0*L/noThreads;

  for spikeNo = 2:noSpikes
    U(realNo+(spikeNo-1)*noReal) = (offset(1)-offset(spikeNo))/speed;
  end

end

%% Third step
for spikeNo = 1:noSpikes

  average = 0.0;

  for realNo = 1:noReal
    average = average+U(realNo+(spikeNo-1)*noReal);
  end

  U(spikeNo) = average/noReal;

end

V = U(1:noSpikes)

%% First step
for realNo = 1:noReal

  count = spikeCount(realNo);

  for spikeNo = 0:noSpikes-1

    val.t   = 0.0;
    val.x   = 0.0;
    val.tSq = 0.0;
    val.xt  = 0.0;
    val.count = 0;

    for i = 1:count

      t = firingTime(i)*(spikeInd(i)==spikeNo);
      x = firingInd(i)*(spikeInd(i)==spikeNo);

      val.t = val.t+t;
      val.x = val.x+x;
      val.count = val.count+(spikeInd(i)==spikeNo);
      val.tSq = val.tSq+t*t;
      val.xt  = val.xt+t*x;

    end

    averages(0*noSpikes*noReal+spikeNo*noReal+realNo) = val.x/count;
    averages(1*noSpikes*noReal+spikeNo*noReal+realNo) = val.t/count;
    averages(2*noSpikes*noReal+spikeNo*noReal+realNo) = val.tSq/count;
    averages(3*noSpikes*noReal+spikeNo*noReal+realNo) = val.xt/count;
    averages(4*noSpikes*noReal+spikeNo*noReal+realNo) = val.count/count;

  end

end

%% Second step
U = zeros(noSpikes*noReal,1);

for realNo = 1:noReal

  xBar = zeros(noSpikes,1);
  tBar = zeros(noSpikes,1);
  speedNum   = 0.0;
  speedDenom = 0.0;
  offset = zeros(noSpikes,1);

  for spikeNo = 1:noSpikes

    noCross       = averages(4*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    xBar(spikeNo) = averages(0*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    tBar(spikeNo) = averages(1*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    tBarSq        = averages(2*noSpikes*noReal+(spikeNo-1)*noReal+realNo);
    xtBar         = averages(3*noSpikes*noReal+(spikeNo-1)*noReal+realNo);

    speedNum    = speedNum+noCross*(xtBar-xBar(spikeNo)*tBar(spikeNo));
    speedDenom  = speedDenom+noCross*(tBarSq-tBar(spikeNo)*tBar(spikeNo));

  end

  speed = speedNum/speedDenom*(2.0*L/noThreads);

  for spikeNo = 1:noSpikes
    offset(spikeNo) = xBar(spikeNo)*2.0*L/noThreads-speed*tBar(spikeNo);
  end

  U(realNo) = speed;

  for spikeNo = 2:noSpikes
    U(realNo+(spikeNo-1)*noReal) = (offset(1)-offset(spikeNo))/speed;
  end

end

%% Third step
for spikeNo = 1:noSpikes

  average = 0.0;

  for realNo = 1:noReal
    average = average+U(realNo+(spikeNo-1)*noReal);
  end

  U(spikeNo) = average/noReal;

end

V = U(1:noSpikes)
