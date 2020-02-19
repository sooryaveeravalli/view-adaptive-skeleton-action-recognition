% bodyinfo = read_skeleton("../nturgb+d_skeletons/S001C001P001R001A001.skeleton");
% save read_skeleton_file.mat bodyinfo

myFolder = ("./nturgb+d_skeletons");
filepattern = fullfile(myFolder, '*.skeleton');
yourfolder = dir(filepattern);
skeletonfiles = {yourfolder.name};
[~,idx] = sort(skeletonfiles);
nooffiles = size(idx,2);
folder = './skeletonfiles';

for i=1:nooffiles
    allbodyinfo = read_skeleton(strcat("./nturgb+d_skeletons/",skeletonfiles(1,i)));
    name = string(skeletonfiles(1,i));
    save(fullfile(folder, sprintf('%s.mat', name)), 'allbodyinfo', '-v6');
    i
end

function bodyinfo = read_skeleton(filename)
% Reads an .skeleton file from "NTU RGB+D 3D Action Recognition Dataset".
% 
% Argrument:
%   filename: full adress and filename of the .skeleton file.
%

fileid = fopen(filename);
framecount = fscanf(fileid,'%d',1); % no of the recorded frames

bodyinfo=[]; % to store multiple skeletons per frame

for f=1:framecount
    bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
    for b=1:bodycount
        clear body;
        body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
        arrayint = fscanf(fileid,'%d',6); % read 6 integers
        body.clipedEdges = arrayint(1);
        body.handLeftConfidence = arrayint(2);
        body.handLeftState = arrayint(3);
        body.handRightConfidence = arrayint(4);
        body.handRightState = arrayint(5);
        body.isResticted = arrayint(6);
        lean = fscanf(fileid,'%f',2);
        body.leanX = lean(1);
        body.leanY = lean(2);
        body.trackingState = fscanf(fileid,'%d',1);
        
        body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
        joints=[];
        for j=1:body.jointCount
            jointinfo = fscanf(fileid,'%f',11);
            joint=[];
            
            % 3D location of the joint j
            joint.x = jointinfo(1);
            joint.y = jointinfo(2);
            joint.z = jointinfo(3);
            
            % 2D location of the joint j in corresponding depth/IR frame
            joint.depthX = jointinfo(4);
            joint.depthY = jointinfo(5);
            
            % 2D location of the joint j in corresponding RGB frame
            joint.colorX = jointinfo(6);
            joint.colorY = jointinfo(7);
            
            % The orientation of the joint j
            joint.orientationW = jointinfo(8);
            joint.orientationX = jointinfo(9);
            joint.orientationY = jointinfo(10);
            joint.orientationZ = jointinfo(11);
            
            % The tracking state of the joint j
            joint.trackingState = fscanf(fileid,'%d',1);
            
            body.joints(j)=joint;
        end
        bodyinfo(f).bodies(b)=body;
    end
end
fclose(fileid);
end
