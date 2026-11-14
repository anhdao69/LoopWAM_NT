import importlib.util
import sys
import unittest

spec = importlib.util.spec_from_file_location('finalizer_under_test', sys.argv.pop(1))
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)

class RendererGuardTests(unittest.TestCase):
    def test_normal_scene_rejections_are_allowed(self):
        finalizer.assert_clean_render_log('Expert plan failed. Success rate: 50/100\n')

    def test_resource_and_renderer_errors_in_completed_cells_are_rejected(self):
        for error in ['OIDN Error: out of memory', 'OIDN Error: changes to the filter are not committed',
                      'torch.OutOfMemoryError: CUDA out of memory', 'cudaErrorMemoryAllocation',
                      'RuntimeError: cannot create buffer', 'ErrorDeviceLost', 'vk::DeviceLostError']:
            with self.subTest(error=error), self.assertRaises(ValueError):
                finalizer.assert_clean_render_log(error + '\nSuccess rate: 100/100')

if __name__ == '__main__':
    unittest.main()
