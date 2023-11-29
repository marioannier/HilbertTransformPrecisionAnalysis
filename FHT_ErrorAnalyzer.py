import numpy as np
import matplotlib.pyplot as plt

class HilbertTransformErrorAnalyzer:
    """
      HilbertTransformErrorAnalyzer - This class is designed to determine the error in the fractional Hilbert
      transform(FHT) when using nonuniformly delayed taps coefficients approach with empirical probabilities.

      Attributes:
          order (float): order ρ of the FHT results in a phase shift of ±ρπ/2 around the central frequency
          number_coef (int): NUmber of coefficients for bulting the FHT.

      Methods:
          __init__(self, order, number_coef):
              Initializes the HilbertTransformErrorAnalyzer for an order using a number_coef.

          calculate_error(self):
              Calculates the error in the fractional Hilbert transform based on the provided coefficients
              and empirical probabilities.

          response_nonuni_HT():

          calculate_rel_rmse():

          error_calculation():

      Example usage:
          >>> order = 1
          >>> number_coef = 7
          >>> analyzer = HilbertTransformErrorAnalyzer(order, number_coef)
          >>> error = analyzer.error_calculation()
          >>> MORE DETAILS; plots....
          >>> print(error)
      """

    def __init__(self, num_sim=1000, err_range=[-0.1, 0.1], order=1, number_coef=7, f_center=8e9):
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
        data_range = np.max(theory_data) - np.min(theory_data)
        # Alternatively, you can use the data range of measured_data:
        # data_range = np.max(measured_data) - np.min(measured_data)

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
        # remove zero coefficients
        # coef_pos = coef_pos[coef_pos != 0]

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
        # the time delay and the coefficient of the th tap in our case at the first channel are given by
        # [1] Y. Dai and J. Yao, “Nonuniformly Spaced Photonic Microwave Delay-Line Filters and Applications,”
        # IEEE Transactions on Microwave Theory and Techniques, vol. 58, no. 11, pp. 3279–3289, Nov. 2010

        ak = np.abs(coef)

        n = len(coef)
        reg_coef = np.arange(-n + 2, n, 2)
        reg_coef = np.insert(reg_coef, n // 2, 0)

        thita = np.where(coef < 0, np.pi, 0)
        tk = reg_coef - thita / (2 * np.pi)

        return ak, tk

    def error_calculation(self, num_sim=100, err_range=[-0.02, 0.02], order=1, number_coef=7, f_center=8e9): # 0.02 is 10% of 2T

        NRMSE = np.zeros((2, num_sim, num_sim))
        angle = np.rad2deg(order * np.pi / 2)

        coef = self.calculate_coef(angle, number_coef)
        ak, tk = self.calculate_non_unif_coef(coef)

        FSR = f_center
        T = 1 / FSR
        f = np.linspace(0e9, 20e9, 3001)

        # the error in time is based on T, the error goes from err_range[0]*T to err_range[0]*T with num_sim steps
        error_limits = np.linspace(err_range[0], err_range[1], num_sim)

        # defining random seed
        # np.random.seed(123)

        tk = tk - tk[0] # - tk[0] is used because the response_nonuni_HT() works only with positive values

        # determining the reference, data without an error
        response_non_ref = -self.response_nonuni_HT(f, ak, T * tk, T)
        response_non_ref_dB = 10 * np.log10(
            np.abs(response_non_ref[300:2200]))  # the bandwidth of interest is delimited here
        response_non_ref_dB = response_non_ref_dB - np.max(response_non_ref_dB)
        response_non_ref_phase = np.angle(response_non_ref[300:2200])  # the bandwidth of interest is delimited here

        for n in range(num_sim):
            print('Simulating: ', n, ' of: ', num_sim)
            for i, e in enumerate(error_limits):
                # adding the error
                tk_rand = np.random.rand(number_coef) * e
                tk_2 = tk + tk_rand - tk[
                    0]  # - tk[0] is used because the response_nonuni_HT() works only with positive values
                #print(tk_rand)
                #print(tk)

                response_non = -self.response_nonuni_HT(f, ak, T * tk_2, T)
                response_non_dB = 10 * np.log10(np.abs(response_non[300:2200]))
                response_non_dB = response_non_dB - np.max(response_non_dB)
                response_non_phase = np.angle(response_non[300:2200])

                '''
                plt.figure()
                plt.plot(f[300:2200], response_non_dB)
                plt.plot(f[300:2200], response_non_ref_dB, '--k', linewidth=1.5)
                plt.legend(['response_non_dB', 'response_non_ref_dB'])
                plt.xlabel('freq')
                plt.ylabel('Amplitude db')
                plt.show()

                plt.figure()
                plt.plot(f[300:2200], response_non_phase)
                plt.plot(f[300:2200], response_non_ref_phase, '--k', linewidth=1.5)
                plt.legend(['response_non_phase', 'response_non_ref_phase'])
                plt.xlabel('freq')
                plt.ylabel('phase db')
                plt.show()
                '''




                NRMSE[0, i, n] = self.calculate_rel_rmse(response_non_ref_dB, response_non_dB)
                NRMSE[1, i, n] = self.calculate_rel_rmse(response_non_ref_phase, response_non_phase)

            #print("Simulation number", n)
            #print("MAGNITUDE:", NRMSE[0, :, n], "PHASE", NRMSE[1, :, n])

        return NRMSE
