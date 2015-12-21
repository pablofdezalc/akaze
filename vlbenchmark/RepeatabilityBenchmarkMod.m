% RepeatabilityBenchmarkMod.m
% Modifications of the RepeatabilityBenchnmark class
% Detect same number of points per detector
% Pixelic location error
% Author: Pablo F. Alcantarilla
% Date: 21-12-2015
%**************************************************************************

classdef RepeatabilityBenchmarkMod < benchmarks.GenericBenchmark ...
    & helpers.Logger & helpers.GenericInstaller

% benchmarks.RepeatabilityBenchmark Image features repeatability
%   benchmarks.RepeatabilityBenchmark('OptionName',optionValue,...) constructs
%   an object to compute the detector repeatability and the descriptor
%   matching scores as given in [1].
%
%   Using this class is a two step process. First, create an instance of the
%   class specifying any parameter needed in the constructor. Then, use
%   obj.testFeatures() to evaluate the scores given a pair of images, the
%   detected features (and optionally their descriptors), and the homography
%   between the two images.
%
%   Use obj.testFeatureExtractor() to evaluate the test for a given detector
%   and pair of images and being able to cache the results of the test.
%
%   DETAILS ON THE REPEATABILITY AND MATCHING SCORES
%
%   The detector repeatability is calculated for two sets of feature frames
%   FRAMESA and FRAMESB detected in a reference image IMAGEA and a second
%   image IMAGEB. The two images are assumed to be related by a known
%   homography H mapping pixels in the domain of IMAGEA to pixels in the
%   domain of IMAGEB (e.g. static camera, no parallax, or moving camera
%   looking at a flat scene). The homography assumes image coordinates with
%   origin in (0,0).
%
%   A perfect co-variant detector would detect the same features in both
%   images regardless of a change in viewpoint (for the features that are
%   visible in both cases). A good detector will also be robust to noise and
%   other distortion. Repeatability is the percentage of detected features
%   that survive a viewpoint change or some other transformation or
%   disturbance in going from IMAGEA to IMAGEB.
%
%   More in detail, repeatability is by default computed as follows:
%
%   1. The elliptical or circular feature frames FRAMEA and FRAMEB,
%      the image sizes SIZEA and SIZEB, and the homography H are
%      obtained.
%
%   2. Features (ellipses or circles) that are fully visible in both
%      images are retained and the others discarded.
%
%   3. For each pair of feature frames A and B, the normalised overlap
%      measure OVERLAP(A,B) is computed. This is defined as the ratio
%      of the area of the intersection over the area of the union of
%      the ellpise/circle FRAMESA(:,A) and FRAMES(:,B) reprojected on
%      IMAGEA by the homography H. Furthermore, after reprojection the
%      size of the ellpises/circles are rescaled so that FRAMESA(:,A)
%      has an area equal to the one of a circle of radius 30 pixels.
%
%   4. Feature are matched optimistically. A candidate match (A,B) is
%      created for every pair of features A,B such that the
%      OVELRAP(A,B) is larger than a certain threshold (defined as 1 -
%      OverlapError) and weighted by OVERLAP(A,B). Then, the final set
%      of matches M={(A,B)} is obtained by performing a greedy
%      bipartite matching between in the weighted graph
%      thus obtained. Greedy means that edges are assigned in order
%      of decreasing overlap.
%
%   5. Repeatability is defined as the ratio of the number of matches
%      M thus obtained and the minimum of the number of features in
%      FRAMESA and FRAMESB:
%
%                                    |M|
%        repeatability = -------------------------.
%                        min(|framesA|, |framesB|)
%
%   RepeatabilityBenchmark can compute the descriptor matching score
%   too (see the 'Mode' option). To define this, a second set of 
%   matches M_d is obtained similarly to the previous method, except 
%   that the descriptors distances are used in place of the overlap, 
%   no threshold is involved in the generation of candidate matches, and 
%   these are selected by increasing descriptor distance rather than by
%   decreasing overlap during greedy bipartite matching. Then the
%   descriptor matching score is defined as:
%
%                              |inters(M,M_d)|
%        matching-score = -------------------------.
%                         min(|framesA|, |framesB|)
%
%   The test behaviour can be adjusted by modifying the following options:
%
%   Mode:: 'Repeatability'
%     Type of score to be calculated. Changes the criteria which are used
%     for finding one-to-one matches between image features.
%
%     'Repeatability'
%       Match frames geometry only. 
%       Corresponds to detector repeatability measure in [1].
%
%     'MatchingScore'
%       Match frames geometry and frame descriptors.
%       Corresponds to detector matching score in [1].
%
%     'DescMatchingScore'
%        Match frames only based on their descriptors.
%
%   OverlapError:: 0.4
%     Maximal overlap error of frames to be considered as
%     correspondences.
%
%   NormaliseFrames:: true
%     Normalise the frames to constant scale (defaults is true for
%     detector repeatability tests, see Mikolajczyk et. al 2005).
%
%   NormalisedScale:: 30
%     When frames scale normalisation applied, fixed scale to which it is
%     normalised to.
%
%   CropFrames:: true
%     Crop the frames out of overlapping regions (regions present in both
%     images).
%
%   Magnification:: 3
%     When frames are not normalised, this parameter is magnification
%     applied to the input frames. Usually is equal to magnification
%     factor used for descriptor calculation.
%
%   WarpMethod:: 'linearise'
%     Numerical method used for warping ellipses. Available mathods are
%     'standard' and 'linearise' for precise reproduction of IJCV2005 
%     benchmark results.
%
%   DescriptorsDistanceMetric:: 'L2'
%     Distance metric used for matching the descriptors. See
%     documentation of vl_alldist2 for details.
%
%   See also: datasets.VggAffineDataset, vl_alldist2
%
%   REFERENCES
%   [1] K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman,
%       J. Matas, F. Schaffalitzky, T. Kadir, and L. Van Gool. A
%       comparison of affine region detectors. IJCV, 1(65):43–72, 2005.

% Authors: Karel Lenc, Andrea Vedaldi

% AUTORIGHTS

  properties
    Opts = struct(...
      'overlapError', 0.4,...
      'normaliseFrames',true,...
      'cropFrames',false,...
      'magnification', 3,...
      'warpMethod', 'linearise',...
      'mode', 'repeatability',...
      'descriptorsDistanceMetric', 'L2',...
      'normalisedScale', 30,...
      'checkLocation',true,...
      'pixelError',5.0,...
      'equalPoints',true,...
      'nmaxPoints',800);
  end

  properties(Constant, Hidden)
    KeyPrefix = 'repeatability';
    %
    Modes = {'repeatability','matchingscore','descmatchingscore'};
    ModesOpts = containers.Map(benchmarks.RepeatabilityBenchmark.Modes,...
      {struct('matchGeometry',true,'matchDescs',false),...
      struct('matchGeometry',true,'matchDescs',true),...
      struct('matchGeometry',false,'matchDescs',true)});
  end

%**************************************************************************
  methods
    function obj = RepeatabilityBenchmarkMod(varargin)
      import benchmarks.*;
      import helpers.*;
      obj.BenchmarkName = 'repeatability';
      if numel(varargin) > 0
        [obj.Opts varargin] = vl_argparse(obj.Opts,varargin);
        obj.Opts.mode = lower(obj.Opts.mode);
        if ~ismember(obj.Opts.mode, obj.Modes)
          error('Invalid mode %s.',obj.Opts.mode);
        end
      end
      varargin = obj.configureLogger(obj.BenchmarkName,varargin);
      obj.checkInstall(varargin);
    end

    %**********************************************************************
    function [score numMatches bestMatches reprojFrames] = ...
        testFeatureExtractor(obj, featExtractor, tf, imageAPath, imageBPath)
      % testFeatureExtractor Image feature extractor repeatability
      %   REPEATABILITY = obj.testFeatureExtractor(FEAT_EXTRACTOR, TF,
      %   IMAGEAPATH, IMAGEBPATH) computes the repeatability REP of a image
      %   feature extractor FEAT_EXTRACTOR and its frames extracted from
      %   images defined by their path IMAGEAPATH and IMAGEBPATH whose
      %   geometry is related by the homography transformation TF.
      %   FEAT_EXTRACTOR must be a subclass of
      %   localFeatures.GenericLocalFeatureExtractor.
      %
      %   [REPEATABILITY, NUMMATCHES] = obj.testFeatureExtractor(...) 
      %   returns also the total number of feature matches found.
      %
      %   [REP, NUMMATCHES, REPR_FRAMES, MATCHES] =
      %   obj.testFeatureExtractor(...) returns cell array REPR_FRAMES which
      %   contains reprojected and eventually cropped frames in
      %   format:
      %
      %   REPR_FRAMES = {CFRAMES_A,CFRAMES_B,REP_CFRAMES_A,REP_CFRAMES_B}
      %
      %   where CFRAMES_A are (cropped) frames detected in the IMAGEAPATH
      %   image REP_CFRAMES_A are CFRAMES_A reprojected to the IMAGEBPATH
      %   image using homography TF. Same hold for frames from the secons
      %   image CFRAMES_B and REP_CFRAMES_B.
      %   MATCHES is an array of size [size(CFRAMES_A),1]. Two frames are
      %   CFRAMES_A(k) and CFRAMES_B(l) are matched when MATCHES(k) = l.
      %   When frame CFRAMES_A(k) is not matched, MATCHES(k) = 0.
      %
      %   This method caches its results, so that calling it again will not
      %   recompute the repeatability score unless the cache is manually
      %   cleared.
      %
      %   See also: benchmarks.RepeatabilityBenchmark().
      import benchmarks.*;
      import helpers.*;

      obj.info('Comparing frames from det. %s and images %s and %s.',...
          featExtractor.Name,getFileName(imageAPath),...
          getFileName(imageBPath));

      imageASign = helpers.fileSignature(imageAPath);
      imageBSign = helpers.fileSignature(imageBPath);
      imageASize = helpers.imageSize(imageAPath);
      imageBSize = helpers.imageSize(imageBPath);
      resultsKey = cell2str({obj.KeyPrefix, obj.getSignature(), ...
        featExtractor.getSignature(), imageASign, imageBSign});
      cachedResults = obj.loadResults(resultsKey);
      nmaxPoints = obj.Opts.nmaxPoints+1;
      
      % When detector does not cache results, do not use the cached data
      if isempty(cachedResults) || ~featExtractor.UseCache
        if obj.ModesOpts(obj.Opts.mode).matchDescs
          [framesA descriptorsA] = featExtractor.extractFeatures(imageAPath);
          [framesB descriptorsB] = featExtractor.extractFeatures(imageBPath);
          
          % Get the same number of points per method assuming that the
          % features are sorted by its feature detector response
          if (obj.Opts.equalPoints == true)
              
              [nrowsA nfeatA] = size(framesA);
              [nrowsB nfeatB] = size(framesB);
                
              framesA(:,nmaxPoints:nfeatA) = [];
              framesB(:,nmaxPoints:nfeatB) = [];
              descriptorsA(:,nmaxPoints:nfeatA) = [];
              descriptorsB(:,nmaxPoints:nfeatB) = [];
          end
          
          [score numMatches bestMatches reprojFrames] = obj.testFeatures(...
            tf, imageASize, imageBSize, framesA, framesB,...
            descriptorsA, descriptorsB);
        else
          [framesA,descriptorsA] = featExtractor.extractFeatures(imageAPath);
          [framesB,descriptorsB] = featExtractor.extractFeatures(imageBPath);
          
          % Get the same number of points per method assuming that the
          % features are sorted by its feature detector response
          if (obj.Opts.equalPoints == true)
              
              [nrowsA nfeatA] = size(framesA);
              [nrowsB nfeatB] = size(framesB);
                
              framesA(:,nmaxPoints:nfeatA) = [];
              framesB(:,nmaxPoints:nfeatB) = [];
          end
          
          [score numMatches bestMatches reprojFrames] = ...
            obj.testFeatures(tf,imageASize, imageBSize,framesA, framesB);
        end
        if featExtractor.UseCache
          results = {score numMatches bestMatches reprojFrames};
          obj.storeResults(results, resultsKey);
        end
      else
        [score numMatches bestMatches reprojFrames] = cachedResults{:};
        obj.debug('Results loaded from cache');
      end

    end

    %**********************************************************************
    function [score numMatches matches reprojFrames] = ...
        testFeatures(obj, tf, imageASize, imageBSize, framesA, framesB, ...
        descriptorsA, descriptorsB)
      % testFeatures Compute repeatability of given image features
      %   [SCORE NUM_MATCHES] = obj.testFeatures(TF, IMAGE_A_SIZE,
      %   IMAGE_B_SIZE, FRAMES_A, FRAMES_B, DESCS_A, DESCS_B) Compute
      %   matching score SCORE between frames FRAMES_A and FRAMES_B
      %   and their descriptors DESCS_A and DESCS_B which were
      %   extracted from pair of images with sizes IMAGE_A_SIZE and
      %   IMAGE_B_SIZE which geometry is related by homography TF.
      %   NUM_MATHCES is number of matches which is calcuated
      %   according to object settings.
      %
      %   [SCORE, NUM_MATCHES, REPR_FRAMES, MATCHES] =
      %   obj.testFeatures(...) returns cell array REPR_FRAMES which
      %   contains reprojected and eventually cropped frames in
      %   format:
      %
      %   REPR_FRAMES = {CFRAMES_A,CFRAMES_B,REP_CFRAMES_A,REP_CFRAMES_B}
      %
      %   where CFRAMES_A are (cropped) frames detected in the IMAGEAPATH
      %   image REP_CFRAMES_A are CFRAMES_A reprojected to the IMAGEBPATH
      %   image using homography TF. Same hold for frames from the secons
      %   image CFRAMES_B and REP_CFRAMES_B.
      %   MATCHES is an array of size [size(CFRAMES_A),1]. Two frames are
      %   CFRAMES_A(k) and CFRAMES_B(l) are matched when MATCHES(k) = l.
      %   When frame CFRAMES_A(k) is not matched, MATCHES(k) = 0.
      import benchmarks.helpers.*;
      import helpers.*;

      obj.info('Computing score between %d/%d frames.',...
          size(framesA,2),size(framesB,2));
      matchGeometry = obj.ModesOpts(obj.Opts.mode).matchGeometry;
      matchDescriptors = obj.ModesOpts(obj.Opts.mode).matchDescs;
      checkLocation = obj.Opts.checkLocation;
      pixelError = obj.Opts.pixelError;
      
      if isempty(framesA) || isempty(framesB)
        matches = zeros(size(framesA,2)); reprojFrames = {};
        obj.info('Nothing to compute.');
        return;
      end
      
      if exist('descriptorsA','var') && exist('descriptorsB','var')
        if size(framesA,2) ~= size(descriptorsA,2) ...
            || size(framesB,2) ~= size(descriptorsB,2)
          obj.error('Number of frames and descriptors must be the same.');
        end
      elseif matchDescriptors
        obj.error('Unable to match descriptors without descriptors.');
      end

      score = 0; numMatches = 0;
      startTime = tic;
      normFrames = obj.Opts.normaliseFrames;
      overlapError = obj.Opts.overlapError;
      overlapThresh = 1 - overlapError;

      % convert frames from any supported format to unortiented
      % ellipses for uniformity
      framesA = localFeatures.helpers.frameToEllipse(framesA) ;
      framesB = localFeatures.helpers.frameToEllipse(framesB) ;

      % map frames from image A to image B and viceversa
      reprojFramesA = warpEllipse(tf, framesA,...
        'Method',obj.Opts.warpMethod) ;
      reprojFramesB = warpEllipse(inv(tf), framesB,...
        'Method',obj.Opts.warpMethod) ;

      % optionally remove frames that are not fully contained in
      % both images
      if obj.Opts.cropFrames
        % find frames fully visible in both images
        bboxA = [1 1 imageASize(2)+1 imageASize(1)+1] ;
        bboxB = [1 1 imageBSize(2)+1 imageBSize(1)+1] ;

        visibleFramesA = isEllipseInBBox(bboxA, framesA ) & ...
          isEllipseInBBox(bboxB, reprojFramesA);

        visibleFramesB = isEllipseInBBox(bboxA, reprojFramesB) & ...
          isEllipseInBBox(bboxB, framesB );

        % Crop frames outside overlap region
        framesA = framesA(:,visibleFramesA);
        reprojFramesA = reprojFramesA(:,visibleFramesA);
        framesB = framesB(:,visibleFramesB);
        reprojFramesB = reprojFramesB(:,visibleFramesB);
        if isempty(framesA) || isempty(framesB)
          matches = zeros(size(framesA,2)); reprojFrames = {};
          return;
        end

        if matchDescriptors
          descriptorsA = descriptorsA(:,visibleFramesA);
          descriptorsB = descriptorsB(:,visibleFramesB);
        end
      end

      if ~normFrames
        % When frames are not normalised, account the descriptor region
        magFactor = obj.Opts.magnification^2;
        framesA = [framesA(1:2,:); framesA(3:5,:).*magFactor];
        reprojFramesB = [reprojFramesB(1:2,:); ...
          reprojFramesB(3:5,:).*magFactor];
      end

      reprojFrames = {framesA,framesB,reprojFramesA,reprojFramesB};
      numFramesA = size(framesA,2);
      numFramesB = size(reprojFramesB,2);

      % Find all ellipse overlaps (in one-to-n array)
      frameOverlaps = fastEllipseOverlap(reprojFramesB, framesA, ...
        'NormaliseFrames',normFrames,'MinAreaRatio',overlapThresh,...
        'NormalisedScale',obj.Opts.normalisedScale);

      matches = [];

      if matchGeometry
        % Create an edge between each feature in A and in B
        % weighted by the overlap. Each edge is a candidate match.
        corresp = cell(1,numFramesA);
        for j=1:numFramesA
          numNeighs = length(frameOverlaps.scores{j});
          if numNeighs > 0
            corresp{j} = [j *ones(1,numNeighs); ...
                          frameOverlaps.neighs{j}; ...
                          frameOverlaps.scores{j}];
          end
        end
        corresp = cat(2,corresp{:}) ;
        if isempty(corresp)
          score = 0; numMatches = 0; matches = zeros(1,numFramesA); return;
        end

        % Remove edges (candidate matches) that have insufficient overlap
        corresp = corresp(:,corresp(3,:) > overlapThresh) ;
        if isempty(corresp)
          score = 0; numMatches = 0; matches = zeros(1,numFramesA); return;
        end

        % Remove correspondences by checking location error
        if checkLocation
           
            [aux ncorr] = size(corresp);
            ids = [];
            
            for ix=1:ncorr
               
                id1 = corresp(1,ix);
                idn = corresp(2,ix);
                x1 = framesA(1,id1);
                y1 = framesA(2,id1);
                xn = framesB(1,idn);
                yn = framesB(2,idn);
                
                s = tf(3,1)*x1+tf(3,2)*y1+tf(3,3);
                x1n = (tf(1,1)*x1+tf(1,2)*y1+tf(1,3))/s;
                y1n = (tf(2,1)*x1+tf(2,2)*y1+tf(2,3))/s;
                
                err = sqrt((x1n-xn)^2 + (y1n-yn)^2);
                if( err > pixelError )
                   ids = [ids, ix]; 
                end
            end
            
            corresp(:,ids) = [];
        end
        
        % Sort the edgest by decrasing score
        [drop, perm] = sort(corresp(3,:), 'descend');
        corresp = corresp(:, perm);

        % Approximate the best bipartite matching
        obj.info('Matching frames geometry.');
        geometryMatches = greedyBipartiteMatching(numFramesA,...
          numFramesB, corresp(1:2,:)');

        matches = [matches ; geometryMatches];
      end

      if matchDescriptors
        obj.info('Computing cross distances between all descriptors');
        dists = vl_alldist2(single(descriptorsA),single(descriptorsB),...
          obj.Opts.descriptorsDistanceMetric);
        obj.info('Sorting distances')
        [dists, perm] = sort(dists(:),'ascend');

        % Create list of edges in the bipartite graph
        [aIdx bIdx] = ind2sub([numFramesA, numFramesB],perm(1:numel(dists)));
        edges = [aIdx bIdx];

        % Find one-to-one best matches
        obj.info('Matching descriptors.');
        descMatches = greedyBipartiteMatching(numFramesA, numFramesB, edges);

        for aIdx=1:numFramesA
          bIdx = descMatches(aIdx);
          [hasCorresp bCorresp] = ismember(bIdx,frameOverlaps.neighs{aIdx});
          % Check whether found descriptor matches fulfill frame overlap
          if ~hasCorresp || ...
             ~frameOverlaps.scores{aIdx}(bCorresp) > overlapThresh
            descMatches(aIdx) = 0;
          end
        end
        matches = [matches ; descMatches];
      end

      % Combine collected matches, i.e. select only equal matches
      validMatches = ...
        prod(single(matches == repmat(matches(1,:),size(matches,1),1)),1);
      matches = matches(1,:) .* validMatches;

      % Compute the score
      numBestMatches = sum(matches ~= 0);
      score = numBestMatches / min(size(framesA,2), size(framesB,2));
      numMatches = numBestMatches;

      obj.info('Score: %g \t Num matches: %g', ...
        score,numMatches);

      obj.debug('Score between %d/%d frames comp. in %gs',size(framesA,2), ...
        size(framesB,2),toc(startTime));
    end

    function signature = getSignature(obj)
      signature = helpers.struct2str(obj.Opts);
    end
  end

  methods (Access = protected)
    function deps = getDependencies(obj)
      deps = {helpers.Installer(),helpers.VlFeatInstaller('0.9.14'),...
        benchmarks.helpers.Installer()};
    end
  end

end

