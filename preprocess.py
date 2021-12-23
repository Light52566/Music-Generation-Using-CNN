"""
CS 464 - Introduction to Machine Learning Course Project

Python code to generate music

@ Authors
Mustafa Yaşar 21702808
Ata Korkusuz
Cemre Biltekin
Muhittin Can Korkut
Kemal Alp Sezer

"""
import json
import music21 as m21
import os
import csv

# Final variables

DATASET_PATH = 'deutschl/erk'
ACCEPTABLE_DURATIONS = [0.25,  # 16th note
                        0.5,  # 8th note
                        0.75,  # Dotted note (8 th note + 16th note)
                        1,  # Quarter note
                        1.5,  # Dotted quarter note
                        2,  # Half note
                        3,  # Three quarter note
                        4]  # Whole note
SAVE_DIRECTORY = 'output'
FINAL_DIRECTORY = 'final_dir'

# us = m21.environment.UserSettings()
# us['musicxmlPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'


def preprocess(path_of_dataset: str) -> None:
    songs = []

    print("Songs are being loaded")
    """
    First of all, as the dataset contains files whose extentions are not 'krn', we need to ignore those files and
    use only files with extension 'krn'.
    """
    for path, _, files in os.walk(path_of_dataset):
        # The for loop finds the dataset by looking at the path_of_dataset and enumerates every path,
        # subdirs (since we don't use them, it has been named as '_', and files)
        for file in files:
            if file[-3:] == "krn":
                """
                m21.converter.parse() function returns the Score of the music file whose path is 'path/file'.
                Score is the entirety of the instrumental and vocal parts of a composition in written form,
                """
                parsed_song = m21.converter.parse(os.path.join(path, file))

                # We will store every songs in the songs array
                # np.append(songs, parsed_song) # Add the parsed song to the songs

                """
                    Music21 durations are measured in Quarter Notes, for instance, an eight note has a 
                    duration of 0.5 and a 1 is a quarter note value. 
                    A Duration represents a span of musical time measurable in terms of quarter notes.
                
                    In order to analyze the songs to determine if they do not contain any unacceptable duration,
                    we use flat function of music21 so that it flats every object in the song into a single list
                    and we use notesAndRests attribute which doesn't contain unnecessary information such as 
                    header, etc.
                    
                    This is done because this operation simplifies the operation of the deep learning model without
                    losing accuracy.
                """
                is_song_acceptable = True
                for element in parsed_song.flat.notesAndRests:
                    if element.duration.quarterLength not in ACCEPTABLE_DURATIONS:
                        is_song_acceptable = False

                if not is_song_acceptable:
                    continue

                """
                    Next step is to transpose the songs to C major and A minor. 
                    If the song is in B major, we should transpose the song by calculating the distance
                    between B (tonic for B major) and C (tonic for C major) and transpose it by using that distance
                    Transposing is done to ensure the fingerings correspond to the same written notes for 
                    any instrument in the family.
                    
                    We don't want to learn about all of the keys because there are 24 keys. Reducing everything 
                    to C major and A minor will ease the work of the model since it will not need to generalize
                    all the 24 keys. By doing that, ultimately we will do substantially less data.
                """
                parsed_song_parts = parsed_song.getElementsByClass(m21.stream.Part)  # Get all the elements that have
                # a given class

                parsed_song_part_measures = parsed_song_parts[0].getElementsByClass(m21.stream.Measure)  # Get all the
                # measures

                extracted_key = parsed_song_part_measures[0][4]  # The key object is stored in the 4th index

                # if the extracted key is not in m21.key.Key, then update it with parsed_song.analyze("key")
                if not isinstance(extracted_key, m21.key.Key):
                    extracted_key = parsed_song.analyze("key")

                # Calculate the distance between the key and the pitch
                if extracted_key.mode == "minor":
                    distance = m21.interval.Interval(extracted_key.tonic, m21.pitch.Pitch("A"))
                elif extracted_key.mode == "major":
                    distance = m21.interval.Interval(extracted_key.tonic, m21.pitch.Pitch("C"))

                transposed_song = parsed_song.transpose(distance)

                songs.append(transposed_song)

    """
    In order to feed the model, the songs must be encoded in an image format.

    Every note must be encoded in the format as follows:
        [Pitch, "_", "_", "_"]
    Number of underscores after the Pitch of the note is determined by the duration of the note.
    For instance, if we have pitch = 60 and duration = 1.0 which is a quarter note, then there must be
    3 underscores afterwards. Each underscore equals to 0.25 duration.
    
    Encode every rest in the song as "r"
    """

    time_step = 0.25

    for i, song in enumerate(songs):
        encoded_song = []
        for element in song.flat.notesAndRests:
            if isinstance(element, m21.note.Note):
                pitch = element.pitch.midi

            elif isinstance(element, m21.note.Rest):
                pitch = "r"  # Encode the rests as 'r'

            number_of_steps = int(element.duration.quarterLength / time_step)

            for s in range(number_of_steps):
                # Add all the steps
                if s != 0:
                    encoded_song.append("_")
                else:
                    encoded_song.append(pitch)

            encoded_song_str = ""
            for e in encoded_song:
                encoded_song_str += str(e) + " "

            """
                Save the encoded song into a single file
            """
            output_path = os.path.join(SAVE_DIRECTORY, str(i))
            with open(output_path, "w") as file:
                file.write(encoded_song_str)

    """
        We have saved every song's encode into their own files, however, we need to have a 
        final file that includes every song's encode.
    """
    end_song_indicator = "/ " * 64  # There are 64 times '/ ' at the end of each song's encode
    final_song = ""
    output_path = os.path.join(SAVE_DIRECTORY)
    for path, _, files in os.walk(output_path):
        for file in files:
            path_of_file = os.path.join(path, file)

            song = None
            with open(path_of_file, "r") as o_file:
                song = o_file.read()
                final_song += song + " " + end_song_indicator

    final_song = final_song[:-1]

    with open(FINAL_DIRECTORY + "/final", "w") as final_file:
        final_file.write(final_song)

    """
        The final_file is composed of pitch numbers, 'r's, and '_'s. However, the model can only understand numbers.
        Therefore, we will map the pitches, 'r's and '_'s into numbers. For instance, 
        
        "72": 0,
        "55: 1,
        ... 
    """
    mapping_dict = {}
    final_song_map = final_song.split()
    key_list = list(set(final_song_map))

    for i, value in enumerate(key_list):
        mapping_dict[value] = i

    with open("mapping/map.json", "w") as mp_file:
        json.dump(mapping_dict, mp_file, indent=4)


if __name__ == '__main__':
    preprocess(DATASET_PATH)
