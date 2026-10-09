import copy
import unittest
from scripts.run_training_only_continuation import remaining_specs, validate_controller, validate_checkpoint, validate_handoff, require_ddp_configs


class ContinuationTests(unittest.TestCase):
    def test_only_unstarted_training_stages_remain(self):
        queues={'concat':[{'label':s} for s in ['concat_long','full_41','full_33','full_dense12']],
                'mix':[{'label':s} for s in ['mix_long','full_14','full_22','full_dense30']]}
        self.assertEqual([s['label'] for s in remaining_specs('concat',queues)],['full_33','full_dense12'])
        self.assertEqual([s['label'] for s in remaining_specs('mix',queues)],['full_22','full_dense30'])
        queues['concat'][1]['label']='wrong'
        with self.assertRaises(ValueError):remaining_specs('concat',queues)

    def test_controller_must_match_user_job_queue_and_exact_output(self):
        argv=['python','scripts/run_kv_campaign.py','--queue','concat','--output-root','/campaign','--phase','run']
        validate_controller(argv,{'SLURM_JOB_ID':'4770'},1025,1025,4770,'concat','/campaign')
        for uid,job,queue,root in [(1000,4770,'concat','/campaign'),(1025,4771,'concat','/campaign'),(1025,4770,'mix','/campaign'),(1025,4770,'concat','/other')]:
            with self.assertRaises(ValueError):validate_controller(argv,{'SLURM_JOB_ID':'4770'},uid,1025,job,queue,root)

    def test_checkpoint_must_contain_final_optimizer_and_both_rng_states(self):
        manifest=dict(planned_updates=21700,planned_windows=2777130,epochs=10,world_size=2,loops=4,action_core_loops=1,version='v0',action_kv_mode='aligned',train_windows=277713,microbatch=8)
        state=dict(update=21700,windows_seen=2777130,epoch=9,next_micro=17358,backend='ddp',rng=[{},{}])
        payload=dict(step=21700,optimizer={'state':{1:{}}},training_state=state,version='v0',video_loops=4,action_loops=1,action_kv_mode='aligned')
        validate_checkpoint(payload,manifest)
        for key,value in [('step',21699),('optimizer',{}),('video_loops',3)]:
            bad=copy.deepcopy(payload);bad[key]=value
            with self.assertRaises(ValueError):validate_checkpoint(bad,manifest)
        bad=copy.deepcopy(payload);bad['training_state']['rng'].pop()
        with self.assertRaises(ValueError):validate_checkpoint(bad,manifest)

    def test_reject_unsupported_resume_backend_before_handoff(self):
        specs=[{'label':'full_33'},{'label':'full_dense12'}]
        configs={'full_33':{'backend':'ddp'},'full_dense12':{'backend':'ddp'}}
        require_ddp_configs(specs,configs)
        configs['full_dense12']['backend']='zero1'
        with self.assertRaises(ValueError):require_ddp_configs(specs,configs)

    def test_handoff_must_be_successful_and_hash_bound(self):
        r=dict(status='complete',job=4770,queue='concat',source_revision='pin',predecessor_root='/old',checkpoint_sha256='hash')
        validate_handoff(r,4770,'concat','pin','/old','hash')
        for key,value in [('status','failed'),('job',4771),('checkpoint_sha256','other'),('source_revision','changed')]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):validate_handoff(bad,4770,'concat','pin','/old','hash')


if __name__=='__main__':unittest.main()
