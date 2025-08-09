# Desctiption
Detecting people in close proximity.


# How to run
1. Create venv
2. Activate venv
```bash
source avenv/bin/activate
```
3. Run the code with desired options
Save model:
```bash
python ./scripts/save_model.py
```

Run on video:
```bash
python main.py --test-file-path=./test_videos/actions2.mpg --save-video
```

Run using camera:
```bash
python main.py
```
