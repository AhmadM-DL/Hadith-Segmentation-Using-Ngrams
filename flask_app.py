from flask import Flask, render_template, request, jsonify
from Source.hadith_predictor import PREDICTOR_INFORMATION_GAIN, get_annotated_hadith, pre_process_hadith, segment_hadith
import random

app = Flask(__name__)
app.config["DEBUG"] = True

hadiths = open("./resources/hadith_samples.txt", encoding="utf8").read().split("\n")

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", hadith= "")

@app.route('/random_hadith', methods=['GET'])
def random_hadith():
    return render_template("index.html", hadith= random.choice(hadiths))

@app.route('/api/v1/segment/', methods=['GET'])
def segment():
    hadith = str(request.args['hadith_text'])
    _, splitpose = segment_hadith(hadith, "./Source/extracted_sanad_maten_lists/sanad_bigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/sanad_unigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/maten_bigrams.npy",
                                          "./Source/extracted_sanad_maten_lists/maten_unigrams.npy",
                                          split_position_predictor=PREDICTOR_INFORMATION_GAIN,
                                          verbose=0)
    hadith = pre_process_hadith(hadith)
    return jsonify({"hadith_text": hadith, "split_position": splitpose})

if __name__ == '__main__':
    try:
        app.run()
    except Exception as ex:
        print(ex)
        input("Dont type anything!")
