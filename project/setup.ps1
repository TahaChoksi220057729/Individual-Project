python -m venv .venv
.\.venv\Scripts\Activate.ps1
echo "VIRTUAL ENVIRONMENT ACTIVATED"
pip install -r ./requirements.txt
$FFMPEG=$(where.exe ffmpeg)
echo $FFMPEG
python main.py $FFMPEG