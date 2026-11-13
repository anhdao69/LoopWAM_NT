# E0 teacher-epoch10 / student-five-epoch RoboTwin-50 evaluation specification

User request: evaluate the published E0-epoch10-teacher-5ep release with [1,1] and [2,2], carefully, using both RTX PRO 6000 servers. User authorizes execution and autonomous queuing. No intervention in other projects.

Checkpoint: anhdao69/LoopWAM_NT at b72f355505d98c473fb91e45d10e16d4cc7ce536, robotwin/Elastic/E0-epoch10-teacher-5ep/epoch_05.pt, SHA256 07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e. Student epoch 5/update237310, T0, parent epoch10/update474620, teacher SHA256 70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea. It is not the older one-epoch E0 student.

Protocol: all50tasks, clean+randomized,100acceptedepisodes/task/setting,seed42;10000episodes per schedule,20000total. Four video loops. Action depths[1,1] or[2,2], bothNFE2 and step condition0.5. Predict32/execute24,14Dqpos,CFG1,FP32weights/BF16compute,original normalization/camera/preprocessing/scene admission/unseen instructions/task horizons/RT32-depth8-OIDN rendering. Preserve unseeded Python instruction selection; do not claim paired trajectories.

Assignment: local RTXPRO6000=[1,1]; server137 RTXPRO6000=[2,2]. Four simulator workers per GPU, shared T5 and CUDA graphs, no server189. Independent detached queues wait for at least60000MiB free before smoke/scoring. Do not kill SimpleMemVLN or other jobs. GPU gates may run with at least12000MiB free; no benchmark throughput inference from contested gates.

Validation: release/LFS/SHA256SUMS checks; source comparison to training revision2c78ff6; verify checkpoint epoch/parent/kind/statistics and frozen teacher tensors; native/graph zero-difference checks on8realinputs for both requested schedules; cross-host output and source equivalence;4-worker smoke on2tasks×2settings×2episodes=8excludedepisodes per assigned schedule. Freeze source/runtime before scored rollout. Final audit requires100cells,10000videos/records per schedule; failed partial attempts excluded. Save comprehensive reports in the requested evaluation_report folder and keep them updated automatically.
