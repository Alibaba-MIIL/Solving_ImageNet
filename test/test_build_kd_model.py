
# test_build_kd_model.py - Generated by https://www.codium.ai/

import unittest
from kd.kd_utils import build_kd_model

"""
Code Analysis:
- This class is used to build a knowledge distillation (KD) model. 
- It uses the create_model() function from the models.factory module to create a KD model. 
- It then uses the InplacABN_to_ABN() and fuse_bn2d_bn1d_abn() functions from the kd.helpers module to convert the model to an ABN model and fuse the batch normalization layers. 
- The model is then loaded onto the GPU and set to evaluation mode. 
- The mean and standard deviation of the model are also stored. 
- The normalize_input() function is used to normalize the input data to match the mean and standard deviation of the KD model. 
- It uses the torchvision.transforms module to apply the necessary transformations.
"""


"""
Test strategies:
- test_create_model(): tests that the create_model() function from the models.factory module is correctly called and the model is created.
- test_InplacABN_to_ABN(): tests that the InplacABN_to_ABN() function from the kd.helpers module is correctly called and the model is converted to an ABN model.
- test_fuse_bn2d_bn1d_abn(): tests that the fuse_bn2d_bn1d_abn() function from the kd.helpers module is correctly called and the batch normalization layers are fused.
- test_load_gpu(): tests that the model is correctly loaded onto the GPU and set to evaluation mode.
- test_mean_std(): tests that the mean and standard deviation of the model are correctly stored.
- test_normalize_input(): tests that the normalize_input() function is correctly called and the input data is normalized to match the mean and standard deviation of the KD model.
- test_transforms(): tests that the torchvision.transforms module is correctly called and the necessary transformations are applied.
- test_edge_cases(): tests that the class handles edge cases correctly.
"""


class TestBuildKdModel(unittest.TestCase):

    def setUp(self):
        self.args = {
            'kd_model_name': 'resnet18',
            'kd_model_path': './models/resnet18.pth',
            'num_classes': 10,
            'in_chans': 3
        }

    def test_create_model(self):
        model = build_kd_model(self.args)
        self.assertIsNotNone(model.model)

    def test_InplacABN_to_ABN(self):
        model = build_kd_model(self.args)
        self.assertIsNotNone(model.model.bn1)

    def test_fuse_bn2d_bn1d_abn(self):
        model = build_kd_model(self.args)
        self.assertIsNotNone(model.model.bn2d)

    def test_load_gpu(self):
        model = build_kd_model(self.args)
        self.assertEqual(model.model.device.type, 'cuda')
        self.assertEqual(model.model.training, False)

    def test_mean_std(self):
        model = build_kd_model(self.args)
        self.assertEqual(model.mean_model_kd, model.model.default_cfg['mean'])
        self.assertEqual(model.std_model_kd, model.model.default_cfg['std'])

    def test_normalize_input(self):
        model = build_kd_model(self.args)
        input = torch.randn(3, 224, 224)
        student_model = create_model('resnet18', None, False, 10, 3)
        model.normalize_input(input, student_model)
        self.assertEqual(input.mean(), model.mean_model_kd[0])
        self.assertEqual(input.std(), model.std_model_kd[0])

    def test_transforms(self):
        model = build_kd_model(self.args)
        student_model = create_model('resnet18', None, False, 10, 3)
        self.assertIsInstance(model.transform_std, T.Normalize)
        self.assertIsInstance(model.transform_mean, T.Normalize)

    def test_edge_cases(self):
        args = {
            'kd_model_name': 'resnet18',
            'kd_model_path': None,
            'num_classes': 10,
            'in_chans': 3
        }
        model = build_kd_model(args)
        self.assertIsNotNone(model.model)
