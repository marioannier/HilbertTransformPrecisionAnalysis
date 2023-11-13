import numpy as np
class HilbertTransformErrorAnalyzer:
    """
      HilbertTransformErrorAnalyzer - Class for analyzing errors in the fractional Hilbert transform.

      This class is designed to determine the error in the fractional Hilbert transform when using
      nonuniformly delayed taps coefficients approach with empirical probabilities.

      Attributes:
          coefficients (list): List of coefficients for the nonuniformly delayed taps approach.
          empirical_probabilities (list): List of empirical probabilities used in the analysis.

      Methods:
          __init__(self, coefficients, empirical_probabilities):
              Initializes the HilbertTransformErrorAnalyzer with coefficients and empirical probabilities.

          calculate_error(self):
              Calculates the error in the fractional Hilbert transform based on the provided coefficients
              and empirical probabilities.

      Example usage:
          >>> coefficients = [0.5, 0.2, -0.1, 0.3]
          >>> probabilities = [0.25, 0.4, 0.2, 0.15]
          >>> analyzer = HilbertTransformErrorAnalyzer(coefficients, probabilities)
          >>> error = analyzer.calculate_error()
          >>> print(error)
      """
    def __init__(self, coefficients, empirical_probabilities):
        # Constructor implementation


    def response_nonuni_HT(f, ak, tk, T):
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
        omega_tau = -2 * np.pi * f[:, np.newaxis] * tk
        extra_phase = 1j * 5.5 * T * 2 * np.pi * f

        response = np.sum(ak * np.exp(omega_tau) * np.exp(extra_phase), axis=1)

        return response



    def response_uni_HT(f, coef, hp, T):
        """
        Calculate the response of a fractional Hilbert transformer based on a uniformly spaced delay-line filter.

        The fractional Hilbert transformer response H_FHT(ω) can be expressed as:
        H_FHT(ω) = ∑_(k=-n)^n β_k * e^(-jωτ_k)

        Reference:
        [Include reference if available]

        Parameters:
            f (numpy.ndarray): Frequency values.
            coef (numpy.ndarray): Coefficients for the uniformly spaced delay-line filter.
            hp (numpy.ndarray): Delay values.
            T (float): Parameter used in the response calculation.

        Returns:
            numpy.ndarray: The calculated response for each frequency in f.
        """
        omega_tau = -2 * np.pi * f[:, np.newaxis] * hp
        extra_phase = 1j * 6 * T * 2 * np.pi * f

        response = np.sum(coef * np.exp(omega_tau) * np.exp(extra_phase), axis=1)

        return response

    def calculate_rel_rmse(theory_data, measured_data):
        """
        Calculate relative root mean square error (rel_rmse) between theory_data and measured_data.

        Parameters:
            theory_data (numpy.ndarray): Theoretical data.
            measured_data (numpy.ndarray): Measured data.

        Returns:
            float: Relative root mean square error.
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

    def error_calculation(num_sim=1000, coef=None, tk=None, err=None):
        NRMSE = np.zeros((2, len(err), num_sim))

        # Fixed computations outside the loop
        ak = np.abs(coef)
        f_center = 8e9
        FSR = f_center
        T = 1 / FSR
        f = np.linspace(0e9, 20e9, 3001)
        w = 2 * np.pi * f
        tk_ref = np.array([0.5, 2.5, 4.5, 7, 9, 11]) - 0.5

        for s in range(num_sim):
            # Precompute random values if they don't change during the loop
            if len(coef) == 6:
                tk_rand = np.random.rand(6) * err
                tk = tk_ref + tk_rand
            else:
                tk = tk_ref

            response_non_ref = -response_nonuni_HT(f, coef, T * tk_ref, T)
            response_non_ref_dB = 10 * np.log10(np.abs(response_non_ref[300:2200]))
            response_non_ref_dB = response_non_ref_dB - np.max(response_non_ref_dB)
            response_non_ref_phase = np.angle(response_non_ref[300:2200])

            response_non = -response_nonuni_HT(f, ak, T * tk, T)
            response_non_dB = 10 * np.log10(np.abs(response_non[300:2200]))
            response_non_dB = response_non_dB - np.max(response_non_dB)
            response_non_phase = np.angle(response_non[300:2200])

            NRMSE[0, :, s] = calculate_rel_rmse(response_non_ref_dB, response_non_dB)
            NRMSE[1, :, s] = calculate_rel_rmse(response_non_ref_phase, response_non_phase)

        return NRMSE


