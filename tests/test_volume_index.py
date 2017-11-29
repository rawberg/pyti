from __future__ import absolute_import
import unittest
import numpy as np

from tests.sample_data import SampleData
from pyti import volume_index


class TestVolumeIndex(unittest.TestCase):
    def setUp(self):
        """Create data to use for testing."""
        self.close_data = SampleData().get_sample_close_data()
        self.volume = SampleData().get_sample_volume()

        self.pvi_expected = [1.0, 1.0131617136727868, 1.0152943403369297,
        1.0220581740172878, 1.0220581740172878, 1.0244368189382491,
        1.0244368189382491, 1.028220752029624, 1.028220752029624,
        1.0248545154752871, 1.0248545154752871, 1.0276690488102527,
        1.0244650041655374, 1.0151167327315447, 1.0151167327315447,
        0.9972084407513031, 0.99256833708695491, 0.99256833708695491,
        0.99517706110413839, 0.97758096185097998, 0.97758096185097998,
        0.97758096185097998, 0.99466607532561435, 0.99006720628561307,
        0.99006720628561307, 0.9885444164271886, 0.9885444164271886,
        0.9885444164271886, 0.97742203212921208, 0.99025749033944777,
        0.99025749033944777, 0.98413804996477638, 0.98413804996477638,
        1.0083488755746211, 1.0195070218621587, 1.0195070218621587,
        1.027593827643883, 1.0150016965753139, 0.9885139274029372,
        0.9885139274029372, 0.99638725967223263, 1.0179472995621497,
        1.0179472995621497, 1.0179472995621497, 1.0152023345418555,
        1.021102146237058, 1.021102146237058, 1.021102146237058,
        1.021102146237058, 1.0180911055175035, 1.0180911055175035,
        1.0111978893126188, 1.0111978893126188, 1.0111978893126188,
        1.0105443684296427, 1.0202089368721159, 1.0202089368721159,
        1.026005793363354, 1.0090436871955135, 1.0090436871955135,
        1.0090436871955135, 1.0110083017906049, 1.0110083017906049,
        1.0110083017906049, 1.0092228233579721, 1.018552576858067,
        1.018552576858067, 1.018552576858067, 1.0341738093020001,
        1.0341738093020001, 1.0203332823118929, 1.0136791827974183,
        1.0136791827974183, 1.016967245623978, 1.016967245623978,
        1.0032973415572677, 1.0032973415572677, 1.0032973415572677,
        0.9903105679265346, 1.0083463579978427, 1.0083463579978427,
        1.0083463579978427, 0.99443202185309887, 0.99443202185309887,
        0.99248437258108435, 0.99508542677016176, 0.99998596364813397,
        0.99671893906281928, 0.99671893906281928, 0.99961601167730751,
        1.0033822060761421, 1.0038356609201491, 1.0038356609201491,
        1.0077272266662285, 1.0077272266662285, 1.0077272266662285,
        1.0137268757771734, 1.0137268757771734, 1.0151713784810255,
        1.0155356443802577, 1.0142669941794833, 1.0142669941794833,
        1.0164083126724668, 1.0164083126724668, 1.0164083126724668,
        1.017934986150671, 1.017934986150671, 1.0057341023553226,
        1.0057341023553226, 1.0006518515247116, 0.99497864129519231,
        0.99497864129519231, 0.99729034581610199, 0.99089375432926652,
        0.99089375432926652, 0.98609032526181195, 0.98609032526181195,
        0.98609032526181195, 0.98609032526181195, 0.97762050893175678,
        0.98168119619126104, 0.97456494228104062, 0.96194062743466069,
        0.96194062743466069, 0.96426638949233867, 0.96426638949233867,
        0.97159869360741902]

        self.nvi_expected = [1.0, 1.0, 1.0, 1.0, 0.99737014309878635,
        0.99737014309878635, 1.0004742987659747, 1.0004742987659747,
        0.99716065719744162, 0.99716065719744162, 1.0043145436667653,
        1.0043145436667653, 1.0043145436667653, 1.0043145436667653,
        1.0062413756294879, 1.0062413756294879, 1.0062413756294879,
        0.98680231838995269, 0.98680231838995269, 0.98680231838995269,
        0.98663450803834407, 1.001543812354337, 1.001543812354337,
        1.001543812354337, 0.9944189699523458, 0.9944189699523458,
        0.9944189699523458, 0.99416561480076548, 0.99416561480076548,
        0.99416561480076548, 0.98635048365765376, 0.98635048365765376,
        0.95843298325586723, 0.95843298325586723, 0.95843298325586723,
        0.98950906327275012, 0.98950906327275012, 0.98950906327275012,
        0.98950906327275012, 0.99088989143683581, 0.99088989143683581,
        0.99088989143683581, 0.99632731175782552, 1.0082109045860519,
        1.0082109045860519, 1.0082109045860519, 1.0161210862361918,
        1.024938792602782, 1.0106146186998928, 1.0106146186998928,
        1.0173304832838312, 1.0173304832838312, 0.99918797201758558,
        0.99641396346902655, 0.99641396346902655, 0.99641396346902655,
        0.99371355483156543, 0.99371355483156543, 0.99371355483156543,
        0.99665482086415003, 0.99435889590200588, 0.99435889590200588,
        0.99137380024919031, 0.99593195460701656, 0.99593195460701656,
        0.99593195460701656, 0.99675568915703361, 0.98682169637474004,
        0.98682169637474004, 0.9880204711359688, 0.9880204711359688,
        0.9880204711359688, 0.98259721740583439, 0.98259721740583439,
        0.98661148727656012, 0.98661148727656012, 0.98443970883380372,
        0.99704600899003337, 0.99704600899003337, 0.99704600899003337,
        1.0034417876586332, 1.0034790446217512, 1.0034790446217512,
        0.99659080642219422, 0.99659080642219422, 0.99659080642219422,
        0.99659080642219422, 0.99659080642219422, 0.99417854456756283,
        0.99417854456756283, 0.99417854456756283, 0.99417854456756283,
        0.9975467383944433, 0.9975467383944433, 1.000864645328748,
        0.99560818602833268, 0.99560818602833268, 0.99695467086972778,
        0.99695467086972778, 0.99695467086972778, 0.99695467086972778,
        0.994176709793965, 0.994176709793965, 0.98225048254896785,
        0.98430800315838363, 0.98430800315838363, 0.98531676754696274,
        0.98531676754696274, 0.9535660945461697, 0.9535660945461697,
        0.9535660945461697, 0.95340247863870453, 0.95340247863870453,
        0.95340247863870453, 0.95677675074268698, 0.95677675074268698,
        0.956510065780109, 0.93419742391107474, 0.93441331173792364,
        0.93441331173792364, 0.93441331173792364, 0.93441331173792364,
        0.93441331173792364, 0.92071826863351425, 0.92071826863351425,
        0.91541969388983713, 0.91541969388983713]

    def test_pvi(self):
        pvi = volume_index.positive_volume_index(self.close_data, self.volume)
        np.testing.assert_array_equal(pvi, self.pvi_expected)

    def test_pvi_invalid_data(self):
        self.close_data.append(1)
        with self.assertRaises(Exception) as cm:
            volume_index.positive_volume_index(self.close_data, self.volume)
        expected = ("Error: mismatched data lengths, check to ensure that all input data is the same length and valid")
        self.assertEqual(str(cm.exception), expected)

    def test_nvi(self):
        nvi = volume_index.negative_volume_index(self.close_data, self.volume)
        np.testing.assert_array_equal(nvi, self.nvi_expected)

    def test_nvi_invalid_data(self):
        self.close_data.append(1)
        with self.assertRaises(Exception) as cm:
            volume_index.negative_volume_index(self.close_data, self.volume)
        expected = ("Error: mismatched data lengths, check to ensure that all input data is the same length and valid")
        self.assertEqual(str(cm.exception), expected)
