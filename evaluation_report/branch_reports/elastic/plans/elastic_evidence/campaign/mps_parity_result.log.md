mps daemon started
--- nvidia-smi 02:06:09
--- nvidia-smi 02:06:55
--- nvidia-smi 02:07:41
--- nvidia-smi 02:08:27
--- nvidia-smi 02:09:13
--- nvidia-smi 02:09:59
runner rc=0
depths equal: True [4, 2, 2, 2, 2, 2, 2, 2, 2, 2]
protocol equal except subset/videos: True
libero_spatial_seed42_task0 episodes 10 successes new/ref 10 10 steps new [74, 76, 74, 75, 80, 74, 78, 76, 76, 74] ref [74, 76, 74, 75, 80, 74, 78, 76, 76, 74]
libero_10_seed42_task0 episodes 10 successes new/ref 10 10 steps new [274, 346, 265, 459, 262, 271, 259, 234, 318, 269] ref [274, 346, 265, 459, 262, 271, 259, 234, 318, 269]
PARITY 20/20 episodes identical (success, policy_steps, queries)
--- mps control log
[2026-10-29 02:11:41.614 Control 4048089] NEW CLIENT 4067648 from user 1228030620: Server already exists
[2026-10-29 02:11:41.625 Control 4048089] Accepting connection...
[2026-10-29 02:11:41.625 Control 4048089] User did not send valid credentials
[2026-10-29 02:11:41.625 Control 4048089] Accepting connection...
[2026-10-29 02:11:41.625 Control 4048089] NEW CLIENT 4067642 from user 1228030620: Server already exists
[2026-10-29 02:12:07.541 Control 4048089] Accepting connection...
[2026-10-29 02:12:07.541 Control 4048089] NEW CLIENT 4067648 from user 1228030620: Server already exists
[2026-10-29 02:12:07.564 Control 4048089] Accepting connection...
[2026-10-29 02:12:07.564 Control 4048089] NEW CLIENT 4067642 from user 1228030620: Server already exists
[2026-10-29 02:17:31.813 Control 4048089] Accepting connection...
[2026-10-29 02:17:31.824 Control 4048089] NEW UI
[2026-10-29 02:17:31.824 Control 4048089] Cmd:quit
[2026-10-29 02:17:31.824 Control 4048089] Try shutting down server 4067306, backoff is disabled
[2026-10-29 02:17:32.014 Control 4048089] Server 4067306 exited with status 0
[2026-10-29 02:17:32.015 Control 4048089] Exit with status 0
--- mps server log
[2026-10-29 02:17:29.882 Server 4067306] Client {PID: 4067648, Context ID: 1} exiting. Number of active client contexts is 0.
[2026-10-29 02:17:30.016 Server 4067306] Server failed to recevie command with status 806, assuming client {PID: 4067648, Context ID: 0} exit
[2026-10-29 02:17:30.016 Server 4067306] Client process 4067648 disconnected
[2026-10-29 02:17:31.031 Server 4067306] Server failed to recevie command with status 806, assuming client {PID: 4048097, Context ID: 0} exit
[2026-10-29 02:17:31.031 Server 4067306] Client process 4048097 disconnected
[2026-10-29 02:17:31.825 Server 4067306] Server has been notified to shutdown
[2026-10-29 02:17:31.825 Server 4067306] Waiting for current clients to finish
[2026-10-29 02:17:31.825 Server 4067306] Exiting
