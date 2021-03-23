# Third party libraries
import numpy as np
import pandas as pd
import pytest

# Internal imports
import significance_testing


class TestSigTester:
    """Testing methods associated with SigTester class."""

    def test__assert_data_is_valid_nps_in_range(self):
        """An appropriate exception is raised if invalid NPS data is provided."""

        mock_data_invalid_nps = pd.DataFrame(
            {
                'respondent': [1, 2, 3],
                'group_id': ['Group 1', 'Group 1', 'Group 2'],
                'nps': [6, 8, 55]  # 55 exceeds expected range of [0, 10]
            }
        )

        with pytest.raises(
                expected_exception=ValueError,
                match=r'The column nps contains data outside of the expected range*'
        ):
            significance_testing.SigTester(data=mock_data_invalid_nps, group_id_column='group_id', nps_column='nps')

    def test__assert_data_is_valid_correct_number_of_groups(self):
        """An appropriate exception is raised if exactly 2 distinct groups are not provided."""

        mock_data_invalid_group_number = pd.DataFrame(
            {
                'respondent': [1, 2, 3],
                'group_id': ['Group 1', 'Group 2', 'Group 3'],  # 3 distinct groups
                'nps': [6, 8, 9],
            }
        )

        with pytest.raises(
                expected_exception=ValueError,
                match=r'The column group_id should only contain 2 groups to compare against one another *'
        ):
            significance_testing.SigTester(
                data=mock_data_invalid_group_number,
                group_id_column='group_id',
                nps_column='nps'
            )

    @pytest.mark.parametrize('significance_level, significant_or_not', [(0.05, False), (0.5, True)])
    def test__determine_p_value_and_significance(self, monkeypatch, significance_level, significant_or_not):
        """The p-value is calculated and evaluated correctly to determine significance."""

        # Mock data where the observed absolute difference in NPS is 100 (group a: 75, group b: -25)
        mock_data = pd.DataFrame(
            {
                'respondent': [1, 2, 3, 4, 5, 6, 7, 8],
                'group_id': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'nps': [0, 10, 10, 10, 0, 0, 7, 10],
            }
        )

        sig_tester = significance_testing.SigTester(data=mock_data, group_id_column='group_id', nps_column='nps')

        # Mock data for repeated random simulations to fix how many are more extreme than what was observed with an
        # Array in which 1 out of 5 is more extreme than observed, which would yield a p-value of 0.2
        mock_simulations = np.array([0, 0, 0, 0, 125])

        monkeypatch.setattr(sig_tester, '_simulated_differences_in_nps', mock_simulations)
        # sig_tester._simulated_differences_in_nps = np.array([0, 0, 0, 0, 125])

        expected_result = (0.2, significant_or_not)

        actual_result = sig_tester._determine_p_value_and_significance(
            observed_difference_in_nps=100,
            significance_level=significance_level
        )

        assert actual_result == expected_result

    @pytest.mark.parametrize('num_parallel_processes', [None, 2])
    def test_repeatedly_simulate_difference_in_nps(self, num_parallel_processes, monkeypatch):
        """
        Simulations are called again and again and their results are stored appropriately, regardless of whether the
        process is parallelised.
        """

        mock_data = pd.DataFrame(
            {
                'respondent': [1, 2, 3, 4, 5, 6, 7, 8],
                'group_id': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'nps': [0, 9, 10, 10, 0, 0, 7, 10],
            }
        )

        sig_tester = significance_testing.SigTester(data=mock_data, group_id_column='group_id', nps_column='nps')

        # The data is randomly shuffled so fix the output to allow us to test it
        def mock_shuffled_nps_responses(*args):
            return np.array([-100, 100, 100, 100,  # Group 1 (first 4 responses) should have NPS of 50
                             -100, -100, 0, 100])  # Group 2 (last 4 responses) should have NPS of -25]

        monkeypatch.setattr(np.random, 'permutation', mock_shuffled_nps_responses)

        num_simulations = 5

        # Expect to see simulations where the absolute difference in NPS is 75 each time (diff between 50 and -25)
        expected_simulated_differences_in_nps = np.array([75] * num_simulations)

        sig_tester.repeatedly_simulate_difference_in_nps(
            num_simulations=num_simulations,
            num_parallel_processes=num_parallel_processes
        )

        assert np.array_equal(sig_tester._simulated_differences_in_nps, expected_simulated_differences_in_nps)

    def test__simulate_difference_in_nps(self, monkeypatch):
        """The absolute difference in NPS between the two groups in a simulated scenario is calculated correctly."""

        mock_data = pd.DataFrame(
            {
                'respondent': [1, 2, 3, 4, 5, 6, 7, 8],
                'group_id': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'nps': [0, 9, 10, 10, 0, 0, 7, 10],
            }
        )

        sig_tester = significance_testing.SigTester(data=mock_data, group_id_column='group_id', nps_column='nps')

        # The data is randomly shuffled so fix the output to allow us to test it
        def mock_shuffled_nps_responses(*args):
            return np.array([-100, 100, 100, 100,  # Group 1 (first 4 responses) should have NPS of 50
                             -100, -100, 0, 100])  # Group 2 (last 4 responses) should have NPS of -25]

        monkeypatch.setattr(np.random, 'permutation', mock_shuffled_nps_responses)

        # Check calculation
        expected_abs_difference_in_nps = 75

        actual_abs_difference_in_nps = sig_tester._simulate_difference_in_nps(num_responses_in_single_group=4)

        assert actual_abs_difference_in_nps == expected_abs_difference_in_nps

    def test__summarise_observed_data(self):
        """Summary stats are correctly calculated on the original data."""

        mock_data = pd.DataFrame(
            {
                'respondent': [1, 2, 3, 4, 5, 6, 7, 8],
                'group_id': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'nps': [9, 10, 9, 0, 3, 7, 9, 10],
            }
        )

        expected_summary = pd.DataFrame(
            {
                'group_id': ['a', 'b'],
                'num_respondents': [4, 4],
                'nps_score': [50.0, 25.0],
                'detractor_percentage': [25, 25],
                'neutral_percentage': [0, 25],
                'promoter_percentage': [75, 50]
            }
        ).set_index('group_id')

        sig_tester = significance_testing.SigTester(data=mock_data, group_id_column='group_id', nps_column='nps')

        sig_tester._summarise_observed_data()

        pd.testing.assert_frame_equal(left=sig_tester.observed_data_summary, right=expected_summary)
