from flask import Flask, render_template

app = Flask(__name__)

import web.config as config

app.config['APP_NAME'] = config.APP_NAME


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

import web.controllers
