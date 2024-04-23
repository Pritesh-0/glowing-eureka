arm = importrobot('r23_arm.urdf');
q0 = homeConfiguration(arm);
ndof = length(q0);

ik = inverseKinematics('RigidBodyTree', arm);
weights = [0, 0, 0, 1, 1, 0];
endEffector = 'roll_link';

for i=1:20000
t = randomConfiguration(arm);
test(i,:) = t;
end

for i = 1:20000
for j = 1:5
t=test(i,j).JointPosition;
vals(i,j)=t;

show(arm,a)
show(arm,b)
