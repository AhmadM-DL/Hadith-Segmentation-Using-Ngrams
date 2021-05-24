from flask import Flask, render_template, request
from Source.hadith_predictor import PREDICTOR_INFORMATION_GAIN, segment_hadith

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html", hadith="حديث")


@app.route('/segment/')
def segment():
    hadith = request.args.get('hadith_text')
    splitpose, splitword = segment_hadith(hadith, "./Source/extracted_sanad_maten_lists/sanad_bigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/sanad_unigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/maten_bigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/maten_unigrams.npy",
                                          split_position_predictor=PREDICTOR_INFORMATION_GAIN,
                                          verbose=0)
    return render_template("index.html", hadith=splitword)



if __name__ == '__main__':
    try:
        app.run()
    except Exception as ex:
        print(ex)
        input("Dont type anything!")
