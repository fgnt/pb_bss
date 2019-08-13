from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
from pathlib import Path
import json

from lazy_dataset.database import DictDatabase


def _get_data():
    from pb_bss import project_root
    from pathlib import Path
    project_root: Path

    cache = (project_root / 'cache')

    cache.mkdir(exist_ok=True)
    if not (cache / 'pb_test_data-master').exists():
        print('Write to', cache)
        url = "https://github.com/fgnt/pb_test_data/archive/master.zip"
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(cache)
        # for line in zipfile.open(file).readlines():
        #     print(line.decode('utf-8'))

    json_path = cache / 'pb_test_data-master' / 'bss_data' / 'bss_data.json'

    text = json_path.read_text()
    text = text.replace('${DB_DIR}', str(cache / 'pb_test_data-master' / 'bss_data'))
    data = json.loads(text)

    import soundfile
    import numpy as np

    def rec_audio_read(obj):
        if isinstance(obj, dict):
            return {
                k: rec_audio_read(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (tuple, list)):
            return np.array([
                rec_audio_read(e)
                for e in obj
            ])
        else:
            data, sample_rate = soundfile.read(obj)
            return data.T

    def read_audio(example):
        example['audio_data'] = rec_audio_read(example['audio_path'])
        return example

    return DictDatabase(data).get_dataset('test').map(read_audio)


def low_reverberation_data():
    """
    >>> import numpy as np
    >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> from IPython.lib.pretty import pprint
    >>> pprint(low_reverberation_data())  # doctest: +ELLIPSIS
    {'audio_path': ...,
     'gender': ['m', 'm'],
     'kaldi_transcription': ["NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
      'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'],
     'log_weights': [1.2027951449295022, -1.2027951449295022],
     'num_samples': {'observation': 38520, 'speech_source': [38520, 46840]},
     'num_speakers': 2,
     'offset': [0, 0],
     'room_dimensions': [[8.196200148369396],
      [6.05928772400428],
      [3.1540068920818385]],
     'sensor_position': ...,
     'snr': 29.749852569493584,
     'sound_decay_time': 0.354,
     'source_decay_time': 0,
     'source_id': ['4k0c0301', '4k6c030t'],
     'source_position': ...,
      [1.6594772807620646, 1.6594772807620646]],
     'speaker_id': ['1272-128104', '1272-128104'],
     'example_id': 'low_reverberation',
     'dataset': 'test',
     'audio_data': {'observation': array(shape=(6, 38520), dtype=float64),
      'speech_image': array(shape=(2, 6, 38520), dtype=float64),
      'speech_reverberation_direct': array(shape=(2, 6, 38520), dtype=float64),
      'speech_source': array(shape=(2, 38520), dtype=float64)}}
    """
    return _get_data()['low_reverberation']


def reverberation_data():
    """

    >>> reverberation_data()

    """
    return _get_data()['reverberation']


if __name__ == '__main__':
    _get_data()