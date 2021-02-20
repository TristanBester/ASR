#!/bin/bash
#readonly LIBRI_TrainSetURL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
readonly_LIBRI_TrainSetURL="https://www.openslr.org/resources/12/train-clean-360.tar.gz"
readonly LIBRI_ValSetURL="https://www.openslr.org/resources/12/test-clean.tar.gz"

echo "Downloading training dataset."
wget $LIBRI_TrainSetURL
echo "Training dataset successfully downloaded."

echo "Downloading validation dataset."
wget $LIBRI_ValSetURL
echo "Validation dataset successfully downloaded."

echo "Extracting datasets."
tar -xvf train-clean-360.tar.gz
tar -xvf test-clean.tar.gz
echo "Databases have successfully been extracted."

echo "Installing dependencies."
#pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torchaudio
pip install  -r requirements.txt

echo "Preprocessing datasets."
python3 -W ignore preprocess_librispeech_dataset.py --root_dir "LibriSpeech" \
                                                    --dataset_name "train-clean-360" \
                                                    --resample "True" \
                                                    --new_sample_rate 8000 \
                                                    --remove_old "True"

python3 -W ignore preprocess_librispeech_dataset.py --root_dir "LibriSpeech" \
                                                    --dataset_name "test-clean" \
                                                    --resample "True" \
                                                    --new_sample_rate 8000 \
                                                    --remove_old "True"
echo "Preprocessing complete."
