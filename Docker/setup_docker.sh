#!/bin/sh


# Macros
REPO_PDF2TXT_URL=https://github.com/adriwitek/pdf2txt





echo 'Starting docker setup script....'


# Python Installation
apt-get update && apt-get install -y software-properties-common 
add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt-get install -y python3.10 python3-pip 
python3 -m pip install --upgrade  pip  --no-cache-dir
apt-get update && apt-get install -y python-is-python3 git git-lfs wget cmake libfreetype6-dev libfontconfig-dev libnss3-dev libjpeg-dev libopenjp2-7 libopenjp2-7-dev ffmpeg libsm6 libxext6 libleptonica-dev tesseract-ocr libtesseract-dev libpoppler-cpp-dev


# Poppler
git clone https://gitlab.freedesktop.org/poppler/poppler.git
# git checkout poppler-0.89.0
cd /app/poppler 
mkdir build 
cd /app/poppler/build 
cmake -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX:PATH=/usr/local     -DENABLE_UNSTABLE_API_ABI_HEADERS=ON     -DBUILD_GTK_TESTS=OFF     -DBUILD_QT5_TESTS=OFF     -DBUILD_CPP_TESTS=OFF     -DENABLE_CPP=ON     -DENABLE_GLIB=OFF     -DENABLE_GOBJECT_INTROSPECTION=OFF     -DENABLE_GTK_DOC=OFF     -DENABLE_QT5=OFF     -DBUILD_SHARED_LIBS=ON  -DENABLE_GPGME=OFF  -DENABLE_LIBTIFF=OFF  -DENABLE_QT6=OFF -DENABLE_LCMS=OFF -DENABLE_LCMS=OFF  -DENABLE_LIBCURL=OFF -DENABLE_BOOST=OFF ..
make 
make install
cd /app
apt-get update && apt-get install poppler-utils


# Tessdata
wget  https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata 
wget  https://github.com/tesseract-ocr/tessdata/raw/main/cat.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata
wget  https://github.com/tesseract-ocr/tessdata/raw/main/eus.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata
wget  https://github.com/tesseract-ocr/tessdata/raw/main/glg.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata




# pdf2txt installation
cd /app
git clone --depth 1 --branch main $REPO_PDF2TXT_URL
cd /app/pdf2txt
python3 -m pip install --use-pep517  --no-cache-dir -r requirements.txt


# Model downloading
mkdir /app/pdf2txt/pipeline/models/nextprocurement_pdfutils
cd /app/pdf2txt/pipeline/models/nextprocurement_pdfutils
git clone --depth 1 https://huggingface.co/BSC-LT/NextProcurement_pdfutils /app/pdf2txt/pipeline/models/nextprocurement_pdfutils/




# MT Model downloading
mkdir /app/pdf2txt/pipeline/models/nllb
cd /app/pdf2txt/pipeline/models/nllb
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/facebook/nllb-200-distilled-600M

cd /app/pdf2txt/pipeline/models/nllb/nllb-200-distilled-600M
wget https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/tokenizer.json?download=true -O tokenizer.json
wget https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model?download=true -O sentencepiece.bpe.model
wget https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin?download=true  -O pytorch_model.bin

#Convert model to ct2 format
cd /app/pdf2txt/pipeline/models/nllb/
ct2-transformers-converter --model nllb-200-distilled-600M --output_dir nllb-ct2




# Lang identification Model downloading
mkdir /app/pdf2txt/pipeline/models/language_identification
wget  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P /app/pdf2txt/pipeline/models/language_identification 





# Cleaning
rm -rf /app/poppler/ && rm -rf /var/cache/apt/archives/



echo 'Done! Setup script execution finished!'
