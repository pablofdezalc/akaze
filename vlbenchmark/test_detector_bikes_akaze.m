% test_detector_bikes_akaze.m
% Script for testing A-KAZE using the vlbenchmark in the Bikes dataset
% Author: Pablo F. Alcantarilla
% Date: 21-12-2015
%**************************************************************************
function test_detector_bikes_akaze()

import datasets.*;
import benchmarks.*;
import localFeatures.*;

%**************************************************************************
featExtractors = {};
featExtractors{end+1} = VlFeatSift('PeakThresh',2);
featExtractors{end+1} = localFeatures.AKAZE('dthreshold',0.0008,'omax',4,'nsublevels',4,'diffusivity',1,'descriptor',3);

%**************************************************************************
dataset = VggAffineDataset('category','bikes');
repBenchmark = RepeatabilityBenchmarkMod('Mode','Repeatability','OverlapError',0.4,'equalPoints',false,'checkLocation',true,'pixelError', 5.0);

repeatability = [];
numCorresp = [];

for d = 1:numel(featExtractors)
 for i = 2:dataset.NumImages
   [repeatability(d,i) numCorresp(d,i)] = ...
     repBenchmark.testFeatureExtractor(featExtractors{d}, ...
                               dataset.getTransformation(i), ...
                               dataset.getImagePath(1), ...
                               dataset.getImagePath(i));
 end
end


detectorNames = cellfun(@(a) a.Name, featExtractors, 'UniformOutput', false);
printScores(detectorNames, 100 * repeatability, 'Repeatability');
printScores(detectorNames, numCorresp, 'Number of correspondences');

figure(1); clf; 
subplot(1,2,1);
plotScores(detectorNames, dataset, 100 * repeatability, 'Repeatability');

subplot(1,2,2);
plotScores(detectorNames, dataset, numCorresp, 'Number of correspondences');

% Save the results
save ubc_names.mat detectorNames;
save ubc_score.mat repeatability;

%**************************************************************************

% --------------------------------------------------------------------
% Helper functions
% --------------------------------------------------------------------

function printScores(detectorNames, scores, name)
  numDetectors = numel(detectorNames);
  maxNameLen = length('Method name');
  for k = 1:numDetectors
    maxNameLen = max(maxNameLen,length(detectorNames{k}));
  end
  fprintf(['\n', name,':\n']);
  formatString = ['%' sprintf('%d',maxNameLen) 's:'];
  fprintf(formatString,'Method name');
  for k = 2:size(scores,2)
    fprintf('\tImg#%02d',k);
  end
  fprintf('\n');
  for k = 1:numDetectors
    fprintf(formatString,detectorNames{k});
    for l = 2:size(scores,2)
      fprintf('\t%6s',sprintf('%.2f',scores(k,l)));
    end
    fprintf('\n');
  end
end

function plotScores(detectorNames, dataset, score, titleText)
  xstart = max([find(sum(score,1) == 0, 1) + 1 1]);
  xend = size(score,2);
  xLabel = dataset.ImageNamesLabel;
  xTicks = dataset.ImageNames;
  plot(xstart:xend,score(:,xstart:xend)','+-','linewidth', 2); hold on ;
  ylabel(titleText) ;
  xlabel(xLabel);
  set(gca,'XTick',xstart:1:xend);
  set(gca,'XTickLabel',xTicks);
  title(titleText);
  set(gca,'xtick',1:size(score,2));
  maxScore = max([max(max(score)) 1]);
  meanEndValue = mean(score(:,xend));
  legendLocation = 'SouthEast';
  if meanEndValue < maxScore/2
    legendLocation = 'NorthEast';
  end
  legend(detectorNames,'Location',legendLocation);
  grid on ;
  axis([xstart xend 0 maxScore]);
end

  function plotMatched(dataset, detector, benchm, imageBIdx)
    [drop drop siftCorresps siftReprojFrames] = ...
      benchm.testFeatureExtractor(detector, ...
                                dataset.getTransformation(imageBIdx), ...
                                dataset.getImagePath(1), ...
                                dataset.getImagePath(imageBIdx));

    % And plot the feature frame correspondences
    subplot(1,2,1);
    imshow(dataset.getImagePath(1));
    benchmarks.helpers.plotFrameMatches(siftCorresps,...
                                        siftReprojFrames,...
                                        'IsReferenceImage',true,...
                                        'PlotMatchLine',false,...
                                        'PlotUnmatched',true);
    subplot(1,2,2);
    imshow(dataset.getImagePath(imageBIdx));
    benchmarks.helpers.plotFrameMatches(siftCorresps,...
                                        siftReprojFrames,...
                                        'IsReferenceImage',false,...
                                        'PlotMatchLine',false,...
                                        'PlotUnmatched',true);
  end

end
