import numpy

def _angle_to_ratation_matrix(rotation_angles):

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


def steering_vector(model_TDOA, stft_size, sampling_rate,
                    model_attenuation=numpy.array(1.),
                    normalize=False):
    frequency = numpy.arange(stft_size//2) * sampling_rate / stft_size
    frequency = frequency[:, None, None]
    model_TDOA = model_TDOA[None, :, :]
    model_mode_vectors = numpy.exp(-2. * numpy.pi * 1j * frequency * model_TDOA)
    model_mode_vectors *= numpy.atleast_3d(model_attenuation)
    if normalize:
        model_mode_vectors /= \
            numpy.sqrt(numpy.sum(
                model_mode_vectors.conj() * model_mode_vectors, axis=1))
    return model_mode_vectors


def get_farfield_TDOA(source_angles, sensor_positions, reference_channel=1,
                      sound_velocity=343):
    """ Calculates the far field time difference of arrival

    :param source_angles: Impinging angle of the planar waves (assumes an
        infinite distance between source and sensor array)
    :type source_angles: 2xK matrix of azimuth and elevation angles.
    :param sensor_positions: Sensor positions
    :type sensor_positions: 3xM matrix, where M is the number of sensors and
        3 are the cartesian dimensions
    :param reference_channel: Reference microphone
    :param sound_velocity: Speed of sound
    :return: Time difference of arrival
    """

    sensors = sensor_positions.shape[1]
    angles = source_angles.shape[1]

    sensor_distance_vector = sensor_positions - \
                             sensor_positions[:, reference_channel - 1, None]
    source_direction_vector = numpy.zeros([3, angles])
    for k in range(angles):
        source_direction_vector[:, k] = numpy.dot(-_angle_to_ratation_matrix(
            source_angles[:, k]
        ), numpy.eye(N=3, M=1))[:, 0]

    projected_distance = numpy.zeros([sensors, angles])
    for s in range(sensors):
        projected_distance[s, :] = numpy.dot(sensor_distance_vector[:, s],
                                             source_direction_vector)

    return projected_distance / sound_velocity


def get_chime_sensor_positions():
    return numpy.array([
        [-10, 0, 10, -10, 0, 10],
        [19, 19, 19, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])/100.
