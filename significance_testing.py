"""Use permutation testing to analyse NPS scores and test for significance."""

# Standard libraries
import multiprocessing as mp
from typing import Tuple

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class SigTester:
    """Helper class for performing significance testing on the NPS responses of two distinct groups."""

    def __init__(self, data: pd.DataFrame, group_id_column: str, nps_column: str):
        """
        Initialise class for running significance test between two sets of NPS scores.

        Parameters
        ----------
        data : pandas DataFrame
            Data containing individual NPS survey responses of two groups
                e.g. | group   | nps |
                     | control | 7   |
                     | test    | 3   |
                     | ...     | ... |
                     | test    | 5   |
        group_id_column : str
            Column in which each respondent's group name is recorded.
        nps_column : str
            Column in which each respondent's NPS response is recorded.
        """

        # Check data is valid e.g. only two groups, NPS scores are in appropriate range
        self._assert_data_is_valid(data=data, group_id_column=group_id_column, nps_column=nps_column)

        # Rename columns so the input data can always be handled the same
        self._data = data.copy()
        self._data.rename(columns={nps_column: 'nps_response', group_id_column: 'group_id'}, inplace=True)

        # Recode the NPS scores so that we can simply take the average across the column to calculate the overall NPS
        self._data.loc[self._data['nps_response'].between(0, 6), 'nps_response_recoded'] = -100  # Detractors
        self._data.loc[self._data['nps_response'].between(7, 8), 'nps_response_recoded'] = 0  # Neutrals
        self._data.loc[self._data['nps_response'].between(9, 10), 'nps_response_recoded'] = 100  # Promoters

        self._nps_responses_recoded = self._data['nps_response_recoded'].values

        # Add a binary flag for detractor, neutral, and promoter status so they can be easily aggregated later
        self._data['detractor'] = np.where(self._data['nps_response_recoded'] == -100, 100, 0)
        self._data['neutral'] = np.where(self._data['nps_response_recoded'] == 0, 100, 0)
        self._data['promoter'] = np.where(self._data['nps_response_recoded'] == 100, 100, 0)

        # Store key stats from the two groups
        self._summarise_observed_data()

        self._observed_difference_in_nps = abs(
            self.observed_data_summary['nps_score'].max() - self.observed_data_summary['nps_score'].min()
        )

        # Placeholder for simulating the difference between the two groups
        self._simulated_differences_in_nps = None

    @staticmethod
    def _assert_data_is_valid(data: pd.DataFrame, group_id_column: str, nps_column: str) -> None:
        """
        Check that the input data is suitable for analysis in that there are only two distinct groups to compare
        against one another, and the NPS responses fall within an appropriate range.

        Parameters
        ----------
        data : pandas DataFrame
            Data containing individual NPS survey responses of two groups
                e.g. | group   | nps |
                     | control | 7   |
                     | test    | 3   |
                     | ...     | ... |
                     | test    | 5   |
        group_id_column : str
            Column in which each respondent's group name is recorded.
        nps_column : str
            Column in which each respondent's NPS response is recorded.

        Raises
        ------
        ValueError
            If some of the NPS responses fall outside the expected range of [0, 10].
            If there are not exactly two groups to compare against one another.
        """

        # NPS scores are always between 0 and 10
        number_of_rows = data.shape[0]

        if data[nps_column].between(0, 10).sum() != number_of_rows:
            raise ValueError(
                f'The column {nps_column} contains data outside of the expected range. NPS scores should be between 0 '
                f'and 10 (inclusive)'
            )

        # Exactly two groups provided to compare against one another
        number_of_unique_groups = data[group_id_column].nunique()

        if number_of_unique_groups != 2:
            raise ValueError(
                f'The column {group_id_column} should only contain 2 groups to compare against one another. The '
                f'data provided contains {number_of_unique_groups}'
            )

    def _determine_p_value_and_significance(
            self,
            observed_difference_in_nps: float,
            significance_level: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Determines whether the observed difference in NPS is significant by comparing it to random simulations to see
        how unusual the result is under the null hypothesis that both groups are the same.

        Parameters
        ----------
        observed_difference_in_nps : float
            Absolute difference in NPS between the two observed groups.
        significance_level : float in interval (0,1), (default 0.05)
            The significance level/probability of a type I error, i.e. likelihood of a false positive (incorrectly
            rejecting the Null Hypothesis when it is in fact true). Default value of 5% is commonly used but you should
            consider what is appropriate given the business context.

        Returns
        -------
        tuple[p-value, significant_or_not]
            p-value : float
                The probability of obtaining results as extreme as what was observed.
            significant_or_not : bool
                True if the observed difference in NPS is significant based upon the random simulations performed. False
                otherwise.
        """

        num_simulations = self._simulated_differences_in_nps.shape[0]

        simulations_greater_than_observed = sum(self._simulated_differences_in_nps >= observed_difference_in_nps)

        p_value = simulations_greater_than_observed / num_simulations

        significant_or_not = p_value < significance_level

        return p_value, significant_or_not

    def repeatedly_simulate_difference_in_nps(self, num_simulations: int, num_parallel_processes: int = None) -> None:
        """
        Randomly shuffle all of the responses again and again and record the difference in NPS on the simulated data.

        Parameters
        ----------
        num_simulations : int
            How many times to repeat the simulations.
        num_parallel_processes : int (default None)
            How many processes to run in parallel.
        """

        print(f'Running {num_simulations:,} simulations')

        num_responses_in_single_group = self.observed_data_summary['num_respondents'].values[0]

        if num_parallel_processes:

            with mp.Pool(num_parallel_processes) as parallel:

                simulated_results = parallel.map(
                        func=self._simulate_difference_in_nps,
                        iterable=[num_responses_in_single_group for _ in range(num_simulations)]
                )

        else:

            simulated_results = [
                self._simulate_difference_in_nps(num_responses_in_single_group) for _ in range(num_simulations)
            ]

        print('Simulations complete')
        self._simulated_differences_in_nps = np.array(simulated_results)

    def summarise_sig_test(self, significance_level: float = 0.05) -> None:
        """
        Analyse simulations and perform test to determine whether the observed difference in NPS between the two groups
        is significant.

        Parameters
        ----------
        significance_level : float in interval (0,1), (default 0.05)
            The significance level/probability of a type I error, i.e. likelihood of a false positive (incorrectly
            rejecting the Null Hypothesis when it is in fact true). Default value of 5% is commonly used but you should
            consider what is appropriate given the business context.
        """

        num_simulations = self._simulated_differences_in_nps.shape[0]

        group_1_name = self.observed_data_summary.index[0]
        group_1_nps = self.observed_data_summary.loc[group_1_name, 'nps_score']
        group_2_name = self.observed_data_summary.index[1]
        group_2_nps = self.observed_data_summary.loc[group_2_name, 'nps_score']

        (p_value, significant_or_not) = self._determine_p_value_and_significance(
            observed_difference_in_nps=self._observed_difference_in_nps,
            significance_level=significance_level
        )

        print(
            f'NPS for the observed data:'
            f'\n\t{group_1_name}: {group_1_nps}'
            f'\n\t{group_2_name}: {group_2_nps}'
            f'\n\tAbsolute difference: {self._observed_difference_in_nps:.2f}'
        )

        print(
            f'\nOut of {num_simulations:,} simulations, {int(p_value * num_simulations):,} displayed a greater '
            f'absolute difference than what was observed.'
        )

        print(f'\nWith a p-value of {p_value} and significance level of {significance_level}, the observed difference '
              f'{"IS" if significant_or_not else "IS NOT"} significant')

    def _simulate_difference_in_nps(self, num_responses_in_single_group) -> float:
        """
        Shuffle all of the responses and reassign them to each group, before calculating the absolute difference in NPS
        between the two groups in this simulated scenario.

        Parameters
        ----------
        num_responses_in_single_group : int
            How many respondents exist in a single group so we can consistently assign the right number of shuffled
            responses to each group.

        Returns
        -------
        float
            The absolute difference in NPS scores between the two groups once all of the responses had been shuffled
            and reassigned to each group.
        """

        randomly_shuffled = np.random.permutation(self._nps_responses_recoded)

        nps_group_1 = randomly_shuffled[:num_responses_in_single_group].mean()
        nps_group_2 = randomly_shuffled[num_responses_in_single_group:].mean()

        return abs(nps_group_2 - nps_group_1)

    def _summarise_observed_data(self) -> None:
        """
        Calculate the observed differences between the two groups and stored a pandas DataFrame summarising the key
        statistics in the observed data for each group.
        """

        self.observed_data_summary = self._data.groupby('group_id').agg(
            num_respondents=pd.NamedAgg(column='nps_response_recoded', aggfunc='count'),
            nps_score=pd.NamedAgg(column='nps_response_recoded', aggfunc='mean'),
            detractor_percentage=pd.NamedAgg(column='detractor', aggfunc='mean'),
            neutral_percentage=pd.NamedAgg(column='neutral', aggfunc='mean'),
            promoter_percentage=pd.NamedAgg(column='promoter', aggfunc='mean'),
        )

    def visualise_stat_test(self) -> None:
        """
        Plot the distribution of randomly simulated differences in NPS and how the observed result compares.
        """

        # Set default plotting parameters
        plt.rcParams['figure.figsize'] = [14, 8]
        plt.rcParams['font.size'] = '18'

        # Plot the distribution of simulated differences in NPS
        density_plot = sns.kdeplot(self._simulated_differences_in_nps, shade=True)
        density_plot.set(
            xlabel='Absolute Difference in NPS Scores Between Groups',
            ylabel='Proportion of Simulations',
            title='How Surprising Is Our Observed Result?'
        )

        # Add a line to show the actual difference observed in the data
        density_plot.axvline(x=self._observed_difference_in_nps, color='red', linestyle='--')
        plt.legend(labels=['Observed Difference', 'Simulated'], loc='upper right')
        plt.show()
