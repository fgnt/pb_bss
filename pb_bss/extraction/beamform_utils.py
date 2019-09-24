import numpy
import numpy as np
from pb_bss.utils import  get_stft_center_frequencies


def _angle_to_rotation_matrix(rotation_angles):
    """
    It would be helpful to explain the meaning of the elevation. Does
    it go from -pi/2 to pi/2 or from 0 to pi?

    Args:
        rotation_angles: Shape (2, K)

    Returns:

    """

    azimuth = rotation_angles[0]
    elevation = rotation_angles[1]

    rotate_y = numpy.asarray([
        [numpy.cos(-elevation), 0, numpy.sin(-elevation)],
        [0, 1, 0],
        [-numpy.sin(-elevation), 0, numpy.cos(-elevation)]
    ])

    rotate_z = numpy.asarray([
        [numpy.cos(azimuth), -numpy.sin(azimuth), 0],
        [numpy.sin(azimuth), numpy.cos(azimuth), 0],
        [0, 0, 1]
    ])

    return numpy.dot(rotate_y, rotate_z)


def get_steering_vector(
        time_difference_of_arrival,
        stft_size=1024,
        sample_rate=16000,
        normalize=False
):
    center_frequencies = get_stft_center_frequencies(stft_size, sample_rate)
    steering_vector = numpy.exp(
        -2j * numpy.pi *
        center_frequencies *
        time_difference_of_arrival[..., numpy.newaxis]
    )
    if normalize:
        steering_vector /= np.linalg.norm(steering_vector, axis=-2, keepdims=True)
    return steering_vector


def get_diffuse_noise_psd(
        sensor_distances,
        fft_size=1024,
        sample_rate=16000,
        sound_velocity=343
):
    """This function does not calculate a PSD matrix.

    Calculates a spatial coherence matrix for a spherically isotropic sound
    field. This can be interpreted as a normalized PSD matrix.

    The implementation is according to Bitzer, Joerg, and K. Uwe Simmer.
    "Superdirective microphone arrays." In Microphone arrays, pp. 19-38.
    Springer, Berlin, Heidelberg, 2001. In particular Equation (2.17).

    Args:
        sensor_distances: Shape (num_channels, num_channels) in meters.
        fft_size:
        sample_rate:
        sound_velocity: Scalar in meters/ second.

    Returns: spatial coherence matrix
        with shape (num_frequency_bins, num_channels, num_channels)

    """
    f = get_stft_center_frequencies(size=fft_size, sample_rate=sample_rate)
    return np.sinc(
        2.0
        * f[:, None, None]
        * sensor_distances[None, :, :]
        / sound_velocity
    )


def get_nearfield_time_of_flight(source_positions, sensor_positions,
                                 sound_velocity=343):
    """ Calculates exact time of flight in seconds without farfield assumption.

    :param source_positions: Array of 3D source position column vectors.
    :param sensor_positions: Array of 3D sensor position column vectors.
    :param sound_velocity: Speed of sound in m/s.
    :return: Time of flight in s.
    """
    # TODO: Check, if this works for any number of sources and sensors.

    assert source_positions.shape[0] == 3
    assert sensor_positions.shape[0] == 3

    difference = source_positions[:, :, None] - sensor_positions[:, None, :]
    difference = numpy.linalg.norm(difference, axis=0)
    return numpy.asarray(difference / sound_velocity)


def get_farfield_time_difference_of_arrival(
        source_angles,
        sensor_positions,
        reference_channel=1,
        sound_velocity=343.,
):
    """ Calculates the far field time difference of arrival

    :param source_angles: Impinging angle of the planar waves (assumes an
        infinite distance between source and sensor array) in radians
    :type source_angles: 2xK matrix of azimuth and elevation angles.
    :param sensor_positions: Sensor positions
    :type sensor_positions: 3xM matrix, where M is the number of sensors and
        3 are the cartesian dimensions
    :param reference_channel: Reference microphone starting from index=0.
    :param sound_velocity: Speed of sound
    :return: Time difference of arrival
    """

    sensors = sensor_positions.shape[1]
    angles = source_angles.shape[1]

    sensor_distance_vector = (
        sensor_positions - sensor_positions[:, reference_channel, None]
    )
    source_direction_vector = numpy.zeros([3, angles])

    for k in range(angles):
        source_direction_vector[:, k] = numpy.dot(
            -_angle_to_rotation_matrix(source_angles[:, k]),
            numpy.eye(N=3, M=1)
        )[:, 0]

    projected_distance = numpy.zeros([sensors, angles])
    for s in range(sensors):
        projected_distance[s, :] = numpy.dot(
            sensor_distance_vector[:, s],
            source_direction_vector
        )

    return projected_distance / sound_velocity


def get_chime_sensor_positions():
    return numpy.array([
        [-10, 0, 10, -10, 0, 10],
        [19, 19, 19, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])/100.
