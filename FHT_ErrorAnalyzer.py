import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class HilbertTransformErrorAnalyzer:
    """
    HilbertTransformErrorAnalyzer - This class is designed to determine the error in the fractional Hilbert
    transform(FHT) when using nonuniformly delayed taps coefficients approach with empirical probabilities.

    Attributes:
        num_sim (int): Number of simulations.
        err_range (list): Range of error rates in taps, given as a percentage, e.g., [-10, 10].
        order (float): Order ρ of the FHT results in a phase shift of ±ρπ/2 around the central frequency.
        number_coef (int): Number of coefficients for building the FHT.
        f_center (float): Central frequency for the response.

    Methods:

        response_nonuni_HT(self, f, ak, tk, T):
            Calculate the response of a fractional Hilbert transformer based on a nonuniformly spaced delay-line filter.

        calculate_rel_rmse(self, theory_data, measured_data):
            Calculate relative root-mean-square error (rel_rmse) between theory_data and measured_data.

        calculate_coef(self, angle=90, numtaps=7):
            Generate coefficients (coef) and magnitudes (ak) for the Fractional Hilbert Transform response.

        calculate_non_unif_coef(self, coef):
            Calculate time delays (tk) and coefficients (ak) for a nonuniformly spaced delay-line filter.

        error_calculation(self, num_sim=100, err_range=[-10, 10], order=1, number_coef=7, f_center=8e9):
            Perform simulations to calculate the NRMSE for magnitude and phase.

        plot_nrmse(self, NRMSE, err_range, num_sim):
            Plot NRMSE distributions for magnitude and phase.

        plot_3d_nrmse(self, NRMSE, err_range, num_sim):
            Plot 3D NRMSE distributions for magnitude and phase.

      Example usage:
            >>> My_FHTErrorAnalyzer = HilbertTransformErrorAnalyzer()
            >>> err_range = [-10, 10]
            >>> num_sim = 1001
            >>> order = 1
            >>> number_coef = 7
            >>> f_center = 8e9
            >>> NRMSE = My_FHTErrorAnalyzer.error_calculation(num_sim, err_range, order, number_coef, f_center)
            >>> My_FHTErrorAnalyzer.plot_nrmse(NRMSE, err_range, num_sim)
            >>> My_FHTErrorAnalyzer.plot_3d_nrmse(NRMSE, err_range, num_sim)
            >>> plt.show()
      """

    def __init__(self, num_sim=1000, err_range=[-10, 10], order=1, number_coef=7, f_center=8e9):
        # Constructor implementation
        self.num_sim = num_sim
        self.err_range = err_range
        self.order = order
        self.number_coef = number_coef
        self.f_center = f_center

    def response_nonuni_HT(self, f, ak, tk, T):
        """
        Calculate the response of a fractional Hilbert transformer based on a nonuniformly spaced delay-line filter.

        The fractional Hilbert transformer response H_FHT(ω) can be expressed as:
        H_FHT(ω) = ∑_(k=-n)^n β_k * e^(-jωτ_k)

        Reference:
        Z. Li, Y. Han, H. Chi, X. Zhang, and J. Yao,
        “A Continuously Tunable Microwave Fractional Hilbert Transformer Based on a Nonuniformly Spaced Photonic Microwave Delay-Line Filter,”
        Journal of Lightwave Technology, vol. 30, no. 12, pp. 1948–1953, Jun. 2012,
        doi: 10.1109/JLT.2012.2191534.

        Parameters:
            f (numpy.ndarray): Frequency values.
            ak (numpy.ndarray): Coefficients for the nonuniformly spaced delay-line filter.
            tk (numpy.ndarray): Delay values.
            T (float): Parameter used in the response calculation.

        Returns:
            numpy.ndarray: The calculated response for each frequency in f.
        """
        response = np.zeros_like(f, dtype=np.complex128)

        for c in range(len(ak)):
            response += ak[c] * np.exp(-1j * tk[c] * 2 * np.pi * f) * np.exp(1j * 5.5 * T * 2 * np.pi * f)
        return response

    def calculate_rel_rmse(self, theory_data, measured_data):
        """
        Calculate relative root-mean-square error (rel_rmse) between theory_data and measured_data.

        Parameters:
            theory_data (numpy.ndarray): Theoretical data.
            measured_data (numpy.ndarray): Measured data.

        Returns:
            float: Relative root-mean-square error.
        """
        # Calculate RMSE
        rmse = np.sqrt(np.mean((theory_data - measured_data) ** 2))

        # Calculate data range
        data_range = np.max(theory_data) - np.min(measured_data)

        # Calculate relative RMSE
        rel_rmse = (rmse / data_range) * 100
        # Alternatively, you can use absolute RMSE without normalization:
        # rel_rmse = rmse
        return rel_rmse

    def calculate_coef(self, angle=90, numtaps=7):
        """
        Generate coefficients (coef) and magnitudes (ak) for the Fractional Hilbert Transform response.

        Parameters:
            angle (float): The value of the angle in degrees.

        Returns:
            tuple: Tuple containing coefficients (coef) and magnitudes (ak).
        """
        # Define the values of alpha
        phi = np.deg2rad(angle)

        # Define the time axis
        t = np.linspace(-6, 6, 5000)

        # Compute the Fractional Hilbert transform
        fht = np.sin(phi) * (2 / np.pi) * (np.sin((np.pi * t) / 2) ** 2) / t
        fht[len(t) // 2] = np.cos(phi)

        # finding the uniformly spaced values
        coef_pos = np.zeros(numtaps // 2)

        for i in range(1, numtaps // 2 + 1):
            td_index = np.argmax(t[len(t) // 2:] >= 2 * (i - 1) + 1)
            coef_pos[i - 1] = fht[len(t) // 2 + td_index]

        # the central coefficient zero for classical HT
        coef_zeroth = np.array(round(abs(fht[len(t) // 2]), 3))

        # Create the symmetric coefficients array
        coef_neg = -coef_pos[0:][::-1]

        # concatenate all coefficients
        coef = np.hstack((coef_neg, coef_zeroth, coef_pos))

        # round values
        coef = np.round(coef, 3)

        return coef

    def calculate_non_unif_coef(self, coef):
        """
        Calculate the time delay (tk) and coefficients (ak) for a nonuniformly spaced delay-line filter based on the input coefficients.

        The time delay and coefficient of the th tap, especially in the first channel, are given by the following reference:
        [1] Y. Dai and J. Yao, “Nonuniformly Spaced Photonic Microwave Delay-Line Filters and Applications,”
        IEEE Transactions on Microwave Theory and Techniques, vol. 58, no. 11, pp. 3279–3289, Nov. 2010.

        Parameters:
            coef (numpy.ndarray): Coefficients for the nonuniformly spaced delay-line filter.

        Returns:
            tuple: Tuple containing magnitudes (ak) and time delays (tk) for each coefficient.
        """

        ak = np.abs(coef)

        n = len(coef)
        reg_coef = np.arange(-n + 2, n, 2)
        reg_coef = np.insert(reg_coef, n // 2, 0)

        thita = np.where(coef < 0, np.pi, 0)
        tk = reg_coef - thita / (2 * np.pi)

        return ak, tk

    def error_calculation(self, num_sim=100, err_range=[-10, 10], order=1, number_coef=7, f_center=8e9):

        """
        Perform simulations to calculate the Normalized Root Mean Square Error (NRMSE) for the fractional Hilbert transform.

        This method simulates the effects of nonuniform delays on the fractional Hilbert transform (FHT) response,
        calculating the NRMSE for both magnitude and phase components.

        Parameters:
            num_sim (int): Number of simulations to perform.
            err_range (list): Range of error percentages for nonuniform delays.
            order (float): Order ρ of the FHT results in a phase shift of ±ρπ/2 around the central frequency.
            number_coef (int): Number of coefficients for building the FHT.
            f_center (float): Central frequency for the fractional Hilbert transform.

        Returns:
            numpy.ndarray: Array containing NRMSE values for both magnitude and phase components.
                Shape: (2, num_sim, num_sim)
        """

        NRMSE = np.zeros((2, num_sim, num_sim))
        angle = np.rad2deg(order * np.pi / 2)

        coef = self.calculate_coef(angle, number_coef)
        ak, tk = self.calculate_non_unif_coef(coef)

        FSR = f_center
        T = 1 / FSR
        f = np.linspace(0e9, 20e9, 3001)

        # transforming err range percentage to number
        err_range = np.array(err_range) * 0.01
        # the error in time is based on T, the error goes from err_range[0]*T to err_range[0]*T with num_sim steps
        error_limits = np.linspace(err_range[0], err_range[1], num_sim)

        tk = tk - tk[0] # - tk[0] is used because the response_nonuni_HT() works only with positive values

        # determining the reference, data without an error
        response_non_ref = -self.response_nonuni_HT(f, ak, T * tk, T)
        response_non_ref_dB = 10 * np.log10(
            np.abs(response_non_ref[300:2200]))  # the bandwidth of interest is delimited here
        response_non_ref_dB = response_non_ref_dB - np.max(response_non_ref_dB)
        response_non_ref_phase = np.angle(response_non_ref[300:2200])  # the bandwidth of interest is delimited here

        for n in range(num_sim):
            print('Simulating: ', n, ' out of ', num_sim , ' simulations')
            for i, e in enumerate(error_limits):
                # adding the error
                tk_rand = np.random.rand(number_coef) * e
                tk_2 = tk + tk_rand - tk[0]  # - tk[0] is used because the response_nonuni_HT() works only with positive values

                response_non = -self.response_nonuni_HT(f, ak, T * tk_2, T)
                response_non_dB = 10 * np.log10(np.abs(response_non[300:2200]))
                response_non_dB = response_non_dB - np.max(response_non_dB)
                response_non_phase = np.angle(response_non[300:2200])


                NRMSE[0, i, n] = self.calculate_rel_rmse(response_non_ref_dB, response_non_dB)
                NRMSE[1, i, n] = self.calculate_rel_rmse(response_non_phase, response_non_ref_phase)

        return NRMSE


    def plot_nrmse(self, NRMSE, err_range, num_sim):
        """
        Plot the distribution of Normalized Root Mean Square Error (NRMSE) for magnitude and phase components.

        This method generates two plots: one for the magnitude NRMSE and another for the phase NRMSE. The plots
        include the distribution of NRMSE for two randomly selected simulations and the mean NRMSE across all simulations.

        Parameters:
            NRMSE (numpy.ndarray): Array containing NRMSE values for both magnitude and phase components.
                Shape: (2, num_sim, num_sim)
            err_range (list): Range of error percentages for nonuniform delays.
            num_sim (int): Number of simulations performed.

        Returns:
            None
        """

        mpl.use('TkAgg')  # using 'TkAgg' to handling figures

        porc = np.linspace(err_range[0], err_range[1], num_sim)
        NRMSE_2d = np.mean(NRMSE, axis=2)
        # getting two random simulations
        rd = np.random.randint(2, num_sim , size=2)

        # Plot for the first case
        plt.figure()
        plt.plot(porc, NRMSE[0, :, rd[0]])
        plt.plot(porc, NRMSE[0, :, rd[1]])
        plt.plot(porc, NRMSE_2d[0, :], '--k', linewidth=1.5)
        plt.legend(['N-RMSE random Case N°1', 'N-RMSE random Case N°2', 'Mean N-RMSE'])
        plt.xlabel('Maximum error rate in all Taps (%)')
        plt.ylabel('N-RMSE(%)')
        plt.title('Magnitude N-RMSE distribution')

        # Plot for the second case
        plt.figure()
        plt.plot(porc, NRMSE[1, :, rd[0]])
        plt.plot(porc, NRMSE[1, :, rd[1]])
        plt.plot(porc, NRMSE_2d[1, :], '--k', linewidth=1.5)
        plt.legend(['N-RMSE random Case N°1', 'N-RMSE random Case N°2', 'Mean N-RMSE'])
        plt.xlabel('Maximum error rate in all Taps (%)')
        plt.ylabel('N-RMSE(%)')
        plt.title('Phase N-RMSE distribution')


    def plot_3d_nrmse(self, NRMSE, err_range, num_sim):
        """
        Plot the 3D distribution of Normalized Root Mean Square Error (NRMSE) for magnitude and phase components.

        This method generates two 3D plots: one for the 3D magnitude NRMSE distribution and another for the 3D phase NRMSE distribution.

        Parameters:
            NRMSE (numpy.ndarray): Array containing NRMSE values for both magnitude and phase components.
                Shape: (2, num_sim, num_sim)
            err_range (list): Range of error percentages for nonuniform delays.
            num_sim (int): Number of simulations performed.

        Returns:
            None
        """

        mpl.use('TkAgg')  # using 'TkAgg' to handling figures
        # Get the dimensions
        dim1, dim2, dim3 = NRMSE.shape

        # Plot for the first dimension
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        x1, y1 = np.meshgrid(np.linspace(err_range[0], err_range[1], num_sim), range(dim3))
        ax1.plot_surface(x1, y1, np.rot90(NRMSE[0, :, :]), cmap='viridis')
        ax1.set_title('3D Magnitude N-RMSE distribution')
        ax1.set_xlabel('Maximum error rate in all Taps (%)')
        ax1.set_ylabel('Simulation number')
        ax1.set_zlabel('Magnitude N-RMSE(%)')

        # Plot for the second dimension
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        x2, y2 = np.meshgrid(np.linspace(err_range[0], err_range[1], num_sim), range(dim3))
        ax2.plot_surface(x2, y2, np.rot90(NRMSE[1, :, :]), cmap='viridis')
        ax2.set_title('3D Phase N-RMSE distribution')
        ax2.set_xlabel('Maximum error rate in all Taps (%)')
        ax2.set_ylabel('Simulation number')
        ax2.set_zlabel('Phase N-RMSE(%)')



