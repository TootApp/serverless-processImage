import os.path
import random
import base64
import time
# import multiprocessing
# import joblib

from rotate_image import rotate_image

sample_size = 100
# cpu_count = multiprocessing.cpu_count()
input_dir = '/Volumes/FILES/us.toot.sesh-questions/'
output_dir = '/Volumes/FILES/us.toot.sesh-questions/processed/'

file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
samples = random.sample(file_names, sample_size)

sum_duration_ns = 0


def process_input(file_name):
    global sum_duration_ns
    start_time_ns = time.time_ns()

    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    with open(input_path, 'rb') as input_image:

        print(f'Reading {file_name}...')
        input_content = input_image.read()

        print(f'Decoding {file_name}...')
        input_data = base64.b64encode(input_content)

        print(f'Processing {file_name}...')
        output_data = rotate_image(input_data)

        print(f'Encoding {file_name}...')
        output_content = base64.b64decode(output_data)

        with open(output_path, 'wb') as output_image:

            print(f'Writing {file_name}...')
            output_image.write(output_content)

            end_time_ns = time.time_ns()
            duration_ns = end_time_ns - start_time_ns
            sum_duration_ns += duration_ns
            print(f'Processed {file_name} in {duration_ns / 1000 / 1000}s')


# print(f'Processing {len(samples)} samples on {cpu_count} CPUs...')
# joblib.Parallel(n_jobs=cpu_count)(joblib.delayed(process_input)(file_name) for file_name in samples)
# print('Done')


for index, file_name in enumerate(samples):
    process_input(file_name)

avg_duration_ns = sum_duration_ns / sample_size
print(f'Average processing time = {avg_duration_ns / 1000 / 1000}s')
