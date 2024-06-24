from flask import Flask
from flask_cors import CORS
from BackEndApp.Controllers.TranslationController import translationController
app = Flask(__name__)
app.register_blueprint(translationController)
CORS(app)

if __name__ == '__main__':
    app.run()