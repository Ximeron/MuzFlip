import os
import soundfile as sf
import hashlib
import argparse

DATASET_PATH = 'DATASET'
SAMPLES_TO_CONSIDER = 22050 * 30  # 30 секунд аудио
HASH_SET = set()

def hash_audio(y):
    """Hash the audio data to check for duplicates."""
    hash_object = hashlib.sha256(y.tobytes())
    return hash_object.hexdigest()

def load_existing_hashes(dataset_path, genres):
    """Load hashes of existing audio files in the dataset."""
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            try:
                with sf.SoundFile(file_path, 'r') as f:
                    y = f.read(dtype='float32')
                    sr = f.samplerate
                    if len(y) >= SAMPLES_TO_CONSIDER:
                        y = y[:SAMPLES_TO_CONSIDER]
                        audio_hash = hash_audio(y)
                        HASH_SET.add(audio_hash)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

def check_and_remove_duplicates(genre_folder):
    """Check for duplicates in the genre folder and remove them."""
    files = os.listdir(genre_folder)
    hashes = {}
    duplicates = []

    for file_name in files:
        file_path = os.path.join(genre_folder, file_name)
        try:
            with sf.SoundFile(file_path, 'r') as f:
                y = f.read(dtype='float32')
                audio_hash = hash_audio(y)

                if audio_hash in hashes:
                    duplicates.append(file_name)
                else:
                    hashes[audio_hash] = file_name
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Remove duplicates
    for duplicate in duplicates:
        duplicate_path = os.path.join(genre_folder, duplicate)
        os.remove(duplicate_path)
        print(f"Removed duplicate: {duplicate}")

def add_to_dataset(input_folder, genre, dataset_path):
    genre_folder = os.path.join(dataset_path, genre)
    os.makedirs(genre_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        try:
            with sf.SoundFile(file_path, 'r') as f:
                y = f.read(dtype='float32')
                sr = f.samplerate

                # Разбиение на 30-секундные фрагменты
                num_samples = len(y)
                for start in range(0, num_samples, SAMPLES_TO_CONSIDER):
                    end = start + SAMPLES_TO_CONSIDER
                    if end > num_samples:
                        break
                    y_segment = y[start:end]

                    # Проверка на дубликаты
                    audio_hash = hash_audio(y_segment)
                    if audio_hash not in HASH_SET:
                        HASH_SET.add(audio_hash)

                        # Сохранение сегмента
                        segment_file_name = f"{genre}_{len(HASH_SET)}.wav"
                        segment_file_path = os.path.join(genre_folder, segment_file_name)
                        sf.write(segment_file_path, y_segment, sr)
                        print(f"Added: {segment_file_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Проверка и удаление дубликатов после добавления всех файлов
    check_and_remove_duplicates(genre_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add music to the dataset.")
    parser.add_argument('input_folder', type=str, help="Path to the folder with new music files.")
    parser.add_argument('genre', type=str, help="Genre of the new music files.")
    args = parser.parse_args()

    GENRES = 'classical jazz pop rock'.split()
    if args.genre not in GENRES:
        print(f"Genre {args.genre} is not recognized. Please use one of the following genres: {', '.join(GENRES)}")
    else:
        # Загрузка хэшей существующих файлов
        load_existing_hashes(DATASET_PATH, GENRES)
        # Добавление новых файлов в датасет
        add_to_dataset(args.input_folder, args.genre, DATASET_PATH)
